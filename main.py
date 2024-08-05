import chainlit as cl
from openai import AsyncOpenAI, OpenAI
import os
from utils import utc_now, speech_to_text, upload_files, process_files
from event_handler import EventHandler

# Configuração de clientes OpenAI
async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
sync_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Recuperar o assistente configurado
assistant = sync_openai_client.beta.assistants.retrieve(
    os.environ.get("OPENAI_ASSISTANT_ID")
)

# Configurar a interface do usuário
config.ui.name = assistant.name

# Inicializar histórico de mensagens
message_history = []

# Evento ao iniciar o chat
@cl.on_chat_start
async def start_chat():
    thread = await async_openai_client.beta.threads.create()
    cl.user_session.set("thread_id", thread.id)
    await cl.Message(content=f"Hello, I'm {assistant.name}!").send()

# Evento ao receber uma mensagem
@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    message_history.append({"role": "user", "content": message.content})
    attachments = await process_files(message.elements)

    # Adicionar uma mensagem à thread
    await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message.content,
        attachments=attachments,
    )

    # Criar e transmitir uma execução
    async with async_openai_client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant.id,
        event_handler=EventHandler(assistant_name=assistant.name),
    ) as stream:
        await stream.until_done()

# Evento ao receber um pedaço de áudio
@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    cl.user_session.get("audio_buffer").write(chunk.data)

# Evento ao finalizar o recebimento de áudio
@cl.on_audio_end
async def on_audio_end(elements: List[Element]):
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    # Adicionar manualmente o nome do arquivo ao BytesIO
    audio_file_buffer = BytesIO(audio_file)
    audio_file_buffer.name = f"input_audio.{audio_mime_type.split('/')[1]}"

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_file_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()

    transcription = await speech_to_text(audio_file_buffer)
    msg = cl.Message(author="You", content=transcription, elements=elements)
    message_history.append({"role": "user", "content": transcription})
    await main(message=msg)

# Configuração do Vector Store para o assistente
async def setup_vector_store():
    try:
        vector_store = await async_openai_client.beta.vector_stores.create(
            name="Product Documentation",
            file_ids=['file_1', 'file_2', 'file_3', 'file_4', 'file_5']
        )
        await async_openai_client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
        )
    except Exception as e:
        print(f"Error setting up vector store: {e}")

# Função principal para iniciar o script
if __name__ == "__main__":
    import asyncio
    asyncio.run(setup_vector_store())
    cl.run()
