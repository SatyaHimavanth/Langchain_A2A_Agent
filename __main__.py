import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, HTTPAuthSecurityScheme
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from dotenv import load_dotenv

from agent import CalculatorAgent
from agent_executor import CalculatorAgentExecutor
from auth import BearerTokenCallContextBuilder
from push_notifications import PushNotificationManager, PushNotificationRouter


load_dotenv()


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)


# Global store for received push notifications (for display in test client)
push_notifications_store = defaultdict(list)  # key: context_id, value: list of notifications


async def handle_push_notification(request):
    """Handle incoming push notifications from agent."""
    try:
        data = await request.json()
        context_id = data.get('contextId', 'unknown')
        
        # Store notification with timestamp
        notification = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            **data
        }
        push_notifications_store[context_id].append(notification)
        
        # Keep only last 50 notifications per context
        if len(push_notifications_store[context_id]) > 50:
            push_notifications_store[context_id] = push_notifications_store[context_id][-50:]
        
        logger.info(f"Received push notification for context {context_id}: {data.get('status', {}).get('state', '?')}")
        
        return JSONResponse({'ok': True, 'message': 'Notification received'})
    except Exception as e:
        logger.error(f"Error handling push notification: {str(e)}")
        return JSONResponse({'ok': False, 'error': str(e)}, status_code=400)


async def handle_get_notifications(request):
    """Get push notifications for a context."""
    try:
        context_id = request.query_params.get('contextId', '')
        if not context_id:
            return JSONResponse({'ok': False, 'error': 'contextId required'}, status_code=400)
        
        notifications = push_notifications_store.get(context_id, [])
        return JSONResponse({
            'ok': True,
            'contextId': context_id,
            'notifications': notifications,
            'count': len(notifications)
        })
    except Exception as e:
        logger.error(f"Error getting notifications: {str(e)}")
        return JSONResponse({'ok': False, 'error': str(e)}, status_code=400)


async def handle_clear_notifications(request):
    """Clear notifications for a context."""
    try:
        context_id = request.query_params.get('contextId', '')
        if context_id in push_notifications_store:
            del push_notifications_store[context_id]
        return JSONResponse({'ok': True, 'message': 'Notifications cleared'})
    except Exception as e:
        logger.error(f"Error clearing notifications: {str(e)}")
        return JSONResponse({'ok': False, 'error': str(e)}, status_code=400)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
@click.option('--timeout', 'timeout', default=60)
@click.option('--agent_url', 'agent_url', default="http://localhost:10000")
def main(host, port, timeout, agent_url):
    """Start the Agent server."""
    try:
        logger.info('Loading model source from `.env` file...')
        model_source = os.getenv('model_source', 'ollama')

        if model_source == 'openai' and not os.getenv('OPENAI_API_KEY'):
            raise MissingAPIKeyError('OPENAI_API_KEY environment variable not set.')
        elif model_source == 'google' and not os.getenv('GOOGLE_API_KEY'):
            raise MissingAPIKeyError('GOOGLE_API_KEY environment variable not set.')
        elif model_source == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
            raise MissingAPIKeyError('ANTHROPIC_API_KEY environment variable not set')
        elif model_source == 'huggingface':
            logger.warning(
                'Please set HUGGINGFACEHUB_API_TOKEN if you are trying to use hosted models'
            )
        elif model_source == 'azure_openai':
            if not os.getenv('AZURE_OPENAI_API_KEY'):
                raise MissingAPIKeyError('AZURE_OPENAI_API_KEY environment variable not set')
            if not os.getenv('AZURE_OPENAI_ENDPOINT'):
                raise MissingAPIKeyError('AZURE_OPENAI_ENDPOINT environment variable not set')
            if not os.getenv('OPENAI_API_VERSION'):
                raise MissingAPIKeyError('OPENAI_API_VERSION environment variable not set')
        else:
            logger.info('Using local ollama by default.')
            logger.info(
                'Please set the model_source in `.env` for custom model provider along with KEYS'
            )

        logger.info('Loading agent url from `.env` file...')
        agent_url = os.getenv('AGENT_URL', agent_url)

        capabilities = AgentCapabilities(streaming=True, push_notifications=True)

        skill = AgentSkill(
            id='calculator_agent',
            name='Basic Calculator',
            description='Helps with basic mathematical calculations',
            tags=['addition', 'subtraction', 'multiplication', 'division'],
            examples=['What is 2+3*5', 'What is 3-3/5*10'],
        )

        extended_skill = AgentSkill(
            id='super_calculator_agent',
            name='Advanced Calculator',
            description='A more advanced mathematical calculator, only for authenticated users.',
            tags=['power', 'root'],
            examples=['What is 2^6/8*1', 'What is sqrt(4)/3*2'],
        )

        public_agent_card = AgentCard(
            name='Calculator Agent',
            description='Basic calculator agent',
            url=agent_url,
            version='1.0.0',
            default_input_modes=CalculatorAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=CalculatorAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            supports_authenticated_extended_card=True,
            security_schemes={
                'bearerAuth': HTTPAuthSecurityScheme(
                    scheme='bearer',
                    bearer_format='Opaque',
                    description='Bearer token required for advanced agent access.',
                )
            },
            skills=[skill],
        )

        extended_agent_card = public_agent_card.model_copy(
            update={
                'name': 'Calculator Agent - Extended Edition',
                'description': 'The full-featured calculator for authenticated users.',
                'version': '1.0.1',
                'security': [{'bearerAuth': []}],
                'skills': [skill, extended_skill],
            }
        )

        def extended_card_modifier(base_card: AgentCard, context):
            if context.user.is_authenticated or context.state.get('auth_token_valid'):
                return base_card
            return public_agent_card

        timeout_config = httpx.Timeout(timeout=max(5.0, timeout))
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        httpx_client = httpx.AsyncClient(
            timeout=timeout_config,
            limits=limits,
            max_redirects=20,
        )
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store,
        )

        # Initialize push notification manager
        push_notification_manager = PushNotificationManager(http_client=httpx_client)
        
        # Create push notification router
        push_notification_router = PushNotificationRouter(push_notification_manager)

        request_handler = DefaultRequestHandler(
            agent_executor=CalculatorAgentExecutor(push_notification_manager=push_notification_manager),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender,
        )

        server = A2AStarletteApplication(
            agent_card=public_agent_card,
            extended_agent_card=extended_agent_card,
            http_handler=request_handler,
            context_builder=BearerTokenCallContextBuilder.from_env(),
            extended_card_modifier=extended_card_modifier,
        )

        app = server.build()
        
        # Add push notification receiver routes
        from starlette.routing import Mount, Route
        routes = app.routes
        # Add routes for receiving and retrieving push notifications
        app.routes.append(Route('/push', handle_push_notification, methods=['POST']))
        app.routes.append(Route('/push/notifications', handle_get_notifications, methods=['GET']))
        app.routes.append(Route('/push/clear', handle_clear_notifications, methods=['POST']))
        
        # Add Push Notification Route Handler Middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        
        class PushNotificationMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Only intercept POST requests to root path
                if request.method == 'POST' and request.url.path == '/':
                    try:
                        body = await request.body()
                        if body:
                            import json
                            data = json.loads(body)
                            method = data.get('method', '')
                            
                            # Check if this is a push notification request
                            if method.startswith('tasks/pushNotificationConfig/'):
                                try:
                                    result = await push_notification_router.handle_request(
                                        method,
                                        data.get('params', {})
                                    )
                                    return JSONResponse({
                                        'jsonrpc': '2.0',
                                        'id': data.get('id'),
                                        'result': result,
                                    })
                                except Exception as e:
                                    logger.error(f"Error handling push notification request: {str(e)}")
                                    return JSONResponse({
                                        'jsonrpc': '2.0',
                                        'id': data.get('id'),
                                        'error': {
                                            'code': -32603,
                                            'message': 'Internal error',
                                            'data': str(e),
                                        }
                                    }, status_code=400)
                    except Exception as e:
                        logger.debug(f"Failed to parse request body for middleware: {str(e)}")
                
                # Let other requests pass through
                return await call_next(request)
        
        # Add middleware in correct order (push notification before CORS)
        app.add_middleware(PushNotificationMiddleware)
        
        # Add CORS middleware to allow all origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allow all HTTP methods
            allow_headers=["*"],  # Allow all headers
        )

        uvicorn.run(app, host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error('Error: %s', e)
        sys.exit(1)
    except Exception as e:
        logger.error('An error occurred during server startup: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
