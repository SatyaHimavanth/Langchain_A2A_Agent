import logging
import os
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.client.middleware import ClientCallContext
from a2a.types import AgentCard, Message, Part, Role, TextPart, TransportProtocol
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from a2a.utils.message import get_message_text
from dotenv import load_dotenv


load_dotenv()


def get_user_query() -> str:
    return input('\n> ')


def _extract_text_from_task_event(task, update) -> str:
    if task and getattr(task, 'artifacts', None):
        return get_message_text(task.artifacts[-1])

    status_update = getattr(update, 'status', None)
    if status_update and getattr(status_update, 'message', None):
        return get_message_text(status_update.message)

    task_status = getattr(task, 'status', None)
    if task_status and getattr(task_status, 'message', None):
        return get_message_text(task_status.message)

    return ''


async def interact_with_server(
    client: Client, call_context: ClientCallContext | None
) -> None:
    context_id = str(uuid4())
    role = Role.user
    print('Conversation started with context_id: ', context_id)

    while True:
        user_input = get_user_query()
        if user_input.lower() == 'exit':
            print('bye!~')
            break

        try:
            message_id = str(uuid4())
            request = Message(
                role=role,
                parts=[Part(TextPart(text=user_input))],
                message_id=message_id,
                context_id=context_id,
            )

            last_artifact_text: str | None = None
            async for response in client.send_message(
                request,
                context=call_context,
            ):
                text = ''
                if isinstance(response, tuple):
                    task, update = response
                    text = _extract_text_from_task_event(task, update)
                else:
                    text = get_message_text(response)

                if text and text != last_artifact_text:
                    print(text)
                    last_artifact_text = text
        except Exception as e:
            print(f'An error occurred: {e}')


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    base_url = 'http://localhost:10000'
    auth_token = os.getenv('A2A_AUTH_TOKEN', 'dummy-token-for-extended-card')
    auth_headers = {'Authorization': f'Bearer {auth_token}'} if auth_token else {}
    if auth_token:
        logger.info('Auth mode: token detected; requesting extended capabilities.')
    else:
        logger.info('Auth mode: no token provided; running as basic client.')
    call_context = (
        ClientCallContext(state={'http_kwargs': {'headers': auth_headers}})
        if auth_headers
        else None
    )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(100.0, connect=100.0),
    ) as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        final_agent_card_to_use: AgentCard | None = None
        try:
            logger.info(
                'Attempting to fetch public agent card from: %s%s',
                base_url,
                AGENT_CARD_WELL_KNOWN_PATH,
            )
            public_card = await resolver.get_agent_card()
            logger.info('Successfully fetched public agent card:')
            logger.info(public_card.model_dump_json(indent=2, exclude_none=True))
            final_agent_card_to_use = public_card
            logger.info('\nUsing PUBLIC agent card for client initialization.')
        except Exception as e:
            logger.error(
                'Critical error fetching public agent card: %s',
                e,
                exc_info=True,
            )
            raise RuntimeError(
                'Failed to fetch the public agent card. Cannot continue.'
            ) from e

        config = ClientConfig(
            httpx_client=httpx_client,
            supported_transports=[
                TransportProtocol.jsonrpc,
                TransportProtocol.http_json,
            ],
            streaming=final_agent_card_to_use.capabilities.streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(final_agent_card_to_use)

        if final_agent_card_to_use.supports_authenticated_extended_card:
            try:
                logger.info(
                    '\nPublic card supports authenticated extended card. '
                    'Attempting to fetch via JSON-RPC method `agent/authenticatedExtendedCard`.'
                )
                extended_card = await client.get_card(context=call_context)
                final_agent_card_to_use = extended_card
                logger.info('Successfully fetched authenticated extended agent card:')
                logger.info(
                    extended_card.model_dump_json(indent=2, exclude_none=True)
                )
                logger.info(
                    '\nUsing AUTHENTICATED EXTENDED agent card for session.'
                )
            except Exception as e:
                logger.warning(
                    'Failed to fetch authenticated extended card: %s. '
                    'Continuing with public card.',
                    e,
                    exc_info=True,
                )
        else:
            logger.info(
                '\nPublic card does not indicate support for an extended card.'
            )

        logger.info('A2AClient initialized.')
        await interact_with_server(client, call_context)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
