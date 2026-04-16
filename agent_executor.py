import logging

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


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


class CalculatorAgentExecutor(AgentExecutor):
    """Decoder Rule Assistant AgentExecutor."""

    def __init__(self):
        self.basic_agent = CalculatorAgent()
        self.advanced_agent = CalculatorAgent(enable_advanced_tools=True)

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
                    terminal_emitted = True
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item['content']))],
                        name='conversion_result',
                    )
                    await updater.complete()
                    terminal_emitted = True
                    break

            if not terminal_emitted:
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        'The request did not produce a terminal response.',
                        task.context_id,
                        task.id,
                    ),
                    final=True,
                )

        except Exception as e:
            logger.error(f'An error occurred while streaming the response: {e}')
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    'An internal error occurred while processing your request.',
                    task.context_id,
                    task.id,
                ),
                final=True,
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
