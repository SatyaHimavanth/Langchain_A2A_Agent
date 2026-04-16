import logging
from typing import Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

from agent import CalculatorAgent
from push_notifications import PushNotificationManager


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


class CalculatorAgentExecutor(AgentExecutor):
    """Decoder Rule Assistant AgentExecutor."""

    def __init__(self, push_notification_manager: Optional[PushNotificationManager] = None):
        self.basic_agent = CalculatorAgent()
        self.advanced_agent = CalculatorAgent(enable_advanced_tools=True)
        self.push_notification_manager = push_notification_manager

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        logging.info('Agent invokation started.')
        logging.info(f'RequestContext: {context.__dict__}')
        logging.info(f'EventQueue: {event_queue.__dict__}')

        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        agent = self._resolve_agent_for_request(context)
        logger.info(
            "Selected '%s' calculator for context_id=%s",
            'advanced' if agent is self.advanced_agent else 'basic',
            context.context_id,
        )

        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            terminal_emitted = False
            async for item in agent.stream(query, task.context_id):
                logging.info(f'Agent response in a2a-protocol: {item}')
                response_status = item.get('status', '')
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                    )
                    # Send push notification for working state
                    if self.push_notification_manager:
                        await self.push_notification_manager.send_task_update(
                            task.id,
                            task.context_id,
                            'working',
                            message=item['content'],
                        )
                elif response_status == 'error':
                    await updater.update_status(
                        TaskState.failed,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    # Send push notification for failed state
                    if self.push_notification_manager:
                        await self.push_notification_manager.send_task_update(
                            task.id,
                            task.context_id,
                            'failed',
                            message=item['content'],
                        )
                    terminal_emitted = True
                    break
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    # Send push notification for input_required state
                    if self.push_notification_manager:
                        await self.push_notification_manager.send_task_update(
                            task.id,
                            task.context_id,
                            'input_required',
                            message=item['content'],
                        )
                    terminal_emitted = True
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item['content']))],
                        name='conversion_result',
                    )
                    await updater.complete()
                    # Send push notification for completed state
                    if self.push_notification_manager:
                        artifacts = [
                            {
                                'parts': [{'text': item['content']}],
                                'name': 'conversion_result',
                            }
                        ]
                        await self.push_notification_manager.send_task_update(
                            task.id,
                            task.context_id,
                            'completed',
                            message=item['content'],
                            artifacts=artifacts,
                        )
                    terminal_emitted = True
                    break

            if not terminal_emitted:
                error_msg = 'The request did not produce a terminal response.'
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        error_msg,
                        task.context_id,
                        task.id,
                    ),
                    final=True,
                )
                # Send push notification for failed state
                if self.push_notification_manager:
                    await self.push_notification_manager.send_task_update(
                        task.id,
                        task.context_id,
                        'failed',
                        message=error_msg,
                    )

        except Exception as e:
            error_msg = 'An internal error occurred while processing your request.'
            logger.error(f'An error occurred while streaming the response: {e}')
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    error_msg,
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
            # Send push notification for failed state
            if self.push_notification_manager:
                await self.push_notification_manager.send_task_update(
                    task.id,
                    task.context_id,
                    'failed',
                    message=f'{error_msg} {str(e)}',
                )

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    def _resolve_agent_for_request(self, context: RequestContext) -> CalculatorAgent:
        if self._is_request_authenticated(context):
            return self.advanced_agent
        return self.basic_agent

    @staticmethod
    def _is_request_authenticated(context: RequestContext) -> bool:
        if not context.call_context:
            return False
        if context.call_context.user.is_authenticated:
            return True
        return bool(context.call_context.state.get('auth_token_valid'))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
