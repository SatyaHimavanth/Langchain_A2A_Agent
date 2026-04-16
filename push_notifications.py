import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

import httpx


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


@dataclass
class PushNotificationConfig:
    """Configuration for push notifications."""
    url: str
    authentication: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return asdict(self)


class PushNotificationManager:
    """Manages push notifications for agent conversations."""

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        """
        Initialize the push notification manager.
        
        Args:
            http_client: Optional httpx.AsyncClient for making HTTP requests
        """
        self.configs: Dict[str, PushNotificationConfig] = {}  # keyed by context_id
        self.http_client = http_client
        self._client_owned = http_client is None

    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=30.0),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
            )

    async def close(self):
        """Close the HTTP client if we own it."""
        if self._client_owned and self.http_client:
            await self.http_client.aclose()

    def set_config(
        self,
        context_id: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Set push notification configuration for a context (conversation).
        
        Args:
            context_id: The context ID (conversation identifier)
            config: Configuration dict with 'url' and optional 'authentication'
            
        Returns:
            Configuration response
        """
        logger.info(f"Setting push config for context {context_id}")
        
        if not config.get('url'):
            raise ValueError("Push notification config must contain 'url'")
        
        push_config = PushNotificationConfig(
            url=config['url'],
            authentication=config.get('authentication'),
        )
        
        self.configs[context_id] = push_config
        
        return {
            'success': True,
            'contextId': context_id,
            'config': push_config.to_dict(),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }

    def get_config(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Get push notification configuration for a context.
        
        Args:
            context_id: The context ID
            
        Returns:
            Configuration dict or None if not found
        """
        config = self.configs.get(context_id)
        if config:
            return config.to_dict()
        return None

    async def send_notification(
        self,
        task_id: str,
        context_id: str,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Send a push notification to the configured webhook.
        
        Args:
            task_id: The task ID
            context_id: The context ID (used to lookup config)
            payload: The notification payload
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        config = self.configs.get(context_id)
        if not config:
            logger.debug(f"No push config found for context {context_id}")
            return False

        await self._ensure_client()

        try:
            headers = {'Content-Type': 'application/json'}
            
            # Add authentication if configured
            if config.authentication:
                auth_config = config.authentication
                if 'credentials' in auth_config:
                    credentials = auth_config['credentials']
                    headers['Authorization'] = f'Bearer {credentials}'
            
            logger.info(f"Sending push notification to {config.url} for task {task_id} (context {context_id})")
            
            # Add metadata to payload
            notification_payload = {
                **payload,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'taskId': task_id,
                'contextId': context_id,
            }
            
            response = await self.http_client.post(
                config.url,
                json=notification_payload,
                headers=headers,
            )
            
            response.raise_for_status()
            logger.info(f"Push notification sent successfully to {config.url}")
            return True
            
        except httpx.RequestError as e:
            logger.error(f"Failed to send push notification: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending push notification: {str(e)}")
            return False

    async def send_task_update(
        self,
        task_id: str,
        context_id: str,
        status: str,
        message: Optional[str] = None,
        artifacts: Optional[list] = None,
    ) -> bool:
        """
        Send a task update notification.
        
        Args:
            task_id: The task ID
            context_id: The context ID (conversation identifier)
            status: Task status (working, completed, failed, etc.)
            message: Optional status message
            artifacts: Optional task artifacts
            
        Returns:
            True if notification was sent successfully
        """
        payload = {
            'taskId': task_id,
            'status': {
                'state': status,
            }
        }
        
        if message:
            payload['status']['message'] = {
                'parts': [{'text': message}]
            }
        
        if artifacts:
            payload['artifacts'] = artifacts
        
        return await self.send_notification(task_id, context_id, payload)


class PushNotificationRouter:
    """Handles JSON-RPC requests for push notifications."""

    def __init__(self, manager: PushNotificationManager):
        """
        Initialize the push notification router.
        
        Args:
            manager: PushNotificationManager instance
        """
        self.manager = manager

    async def handle_set_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle tasks/pushNotificationConfig/set request.
        Configuration is set per context (conversation), not per individual task.
        
        Args:
            params: Request parameters with 'contextId' and 'pushNotificationConfig'
            
        Returns:
            Response dict
        """
        # Support both 'id' (legacy) and 'contextId' (preferred)
        context_id = params.get('contextId') or params.get('id')
        config = params.get('pushNotificationConfig')
        
        if not context_id:
            raise ValueError("Missing 'contextId' parameter")
        if not config:
            raise ValueError("Missing 'pushNotificationConfig' parameter")
        
        result = self.manager.set_config(context_id, config)
        return result

    async def handle_get_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle tasks/pushNotificationConfig/get request.
        
        Args:
            params: Request parameters with 'contextId'
            
        Returns:
            Response dict
        """
        # Support both 'id' (legacy) and 'contextId' (preferred)
        context_id = params.get('contextId') or params.get('id')
        
        if not context_id:
            raise ValueError("Missing 'contextId' parameter")
        
        config = self.manager.get_config(context_id)
        
        if not config:
            return {
                'success': False,
                'contextId': context_id,
                'message': f'No push config found for context {context_id}',
            }
        
        return {
            'success': True,
            'contextId': context_id,
            'config': config,
        }

    async def handle_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route push notification requests.
        
        Args:
            method: The JSON-RPC method name
            params: The request parameters
            
        Returns:
            Response dict
            
        Raises:
            ValueError: If the method is not recognized
        """
        if method == 'tasks/pushNotificationConfig/set':
            return await self.handle_set_config(params)
        elif method == 'tasks/pushNotificationConfig/get':
            return await self.handle_get_config(params)
        else:
            raise ValueError(f"Unknown push notification method: {method}")
