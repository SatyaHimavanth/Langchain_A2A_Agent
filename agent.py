import os
import httpx
import logging

from collections.abc import AsyncIterable
from typing import Any, Literal
from pydantic import BaseModel

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.structured_output import ProviderStrategy

from llm_models import llm
from agent_tools import addition, division, multiplication, power, root, subtraction

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


memory = MemorySaver()


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class CalculatorAgent:
    """Calculator Agent"""

    SYSTEM_INSTRUCTION = (
        'You are a helpful and precise mathematical assistant. '
        'Your primary function is to solve arithmetic and mathematical problems accurately '
        'Use available tools to resolved the users mathematical questions '
    )

    FORMAT_INSTRUCTION = (
        'Set response status to input_required if the user needs to provide more information to complete the request.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    )

    def __init__(self, enable_advanced_tools: bool = False):
        self.model = llm

        self.tools = [addition, subtraction, multiplication, division]
        if enable_advanced_tools:
            self.tools.extend([power, root])

        self.graph = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            system_prompt=self.SYSTEM_INSTRUCTION,
            # Use below if model provider doesn't supports native structured output
            response_format=ToolStrategy(ResponseFormat),

            # Use below if model provider supports native structured output
            # response_format=ProviderStrategy(ContactInfo)
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}
        logger.info(f"Agent invoked with thread_id: {context_id}")

        for chunk in self.graph.stream(inputs, config, stream_mode='updates'):
            logger.info("Agent response:")
            logger.info(f"chunk: {chunk}")

            for step, data in chunk.items():
                if structured_response := data.get('structured_response'):
                    yield self.get_agent_response(structured_response)
                    continue

                message = data['messages'][-1]

                if isinstance(message, AIMessage):
                    if message.tool_calls and len(message.tool_calls) > 0:
                        for tool in message.tool_calls:
                            tool_name = tool["name"]
                            tool_args = str(tool["args"])
                            yield {
                                'status': 'working',
                                'is_task_complete': False,
                                'require_user_input': False,
                                'content': f'Invoking {tool_name} tool with arguments {tool_args}',
                            }
                    else:
                        # No structured terminal payload from the model; treat as
                        # regular completion if content exists, else ask for input.
                        has_content = bool(str(message.content).strip())
                        yield {
                            'status': 'completed' if has_content else 'input_required',
                            'is_task_complete': has_content,
                            'require_user_input': not has_content,
                            'content': message.content or 'Please provide more details.',
                        }
                elif isinstance(message, ToolMessage):
                    tool_name = message.name
                    tool_response = message.content
                    yield {
                        'status': 'working',
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': f'Response from {tool_name} tool is {tool_response}',
                    }


    def get_agent_response(self, structured_response):
        logging.info(f'Final agent response: {structured_response}')
        logging.info(f'Final response type: {type(structured_response)}')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'status': 'input_required',
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'status': 'error',
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'status': 'completed',
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'status': 'error',
            'is_task_complete': False,
            'require_user_input': False,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
