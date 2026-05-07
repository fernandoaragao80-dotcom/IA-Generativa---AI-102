import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speech_sdk

# Variável global para a configuração do serviço de fala
speech_config = None


def ouvir_do_microfone():
    """
    Escuta o microfone e retorna o que foi dito como uma string de texto.
    """
    global speech_config
    if not speech_config:
        print("Erro: Configuração de fala não inicializada.")
        return ""

    # Configura o áudio para usar o microfone padrão
    audio_config = speech_sdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config)

    print("\nOuvindo... (pode falar)")
    # 'recognize_once_async' aguarda o fim da fala para processar
    speech_result = speech_recognizer.recognize_once_async().get()

    if speech_result.reason == speech_sdk.ResultReason.RecognizedSpeech:
        return speech_result.text
    elif speech_result.reason == speech_sdk.ResultReason.NoMatch:
        print("Não entendi o áudio.")
        return ""
    else:
        print(f"Erro no reconhecimento: {speech_result.reason}")
        return ""

def falar_texto(texto_para_falar):
    """
    Transforma texto em áudio (Síntese de voz).
    """
    global speech_config
    if not speech_config: return

    audio_config = speech_sdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_synthesizer = speech_sdk.SpeechSynthesizer(speech_config, audio_config)
    speech_synthesizer.speak_text_async(texto_para_falar).get()

def main(): 
    global speech_config
    try: 
        load_dotenv()
        
        # Configurações Azure OpenAI
        azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
        azure_oai_key = os.getenv("AZURE_OAI_KEY")
        azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
        
        # Configurações Speech Service
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        speech_region = os.getenv("AZURE_SPEECH_REGION")

        # 1. Inicializar Cliente OpenAI
        client = AzureOpenAI(
            azure_endpoint=azure_oai_endpoint, 
            api_key=azure_oai_key, 
            api_version="2024-12-01-preview"
        )

        # 2. Inicializar Configuração de Fala (STT e TTS)
        speech_config = speech_sdk.SpeechConfig(speech_key, speech_region)
        speech_config.speech_recognition_language = "pt-BR"
        speech_config.speech_synthesis_voice_name = "pt-BR-FranciscaNeural"
        
        messages_array = [{"role": "system", "content": "Você é um assistente virtual que responde de forma breve."}]

        print("--- Assistente por Voz Iniciado ---")
        print("Diga algo para começar ou 'Sair' para encerrar.")

    except Exception as ex:
        print(f"Erro na inicialização: {ex}")
        return

    while True:
        # 3. RECONHECIMENTO DE FALA (Substituindo o input de teclado)
        input_text = ouvir_do_microfone()
        
        if not input_text:
            continue
            
        print(f"Você disse: {input_text}")
        
        if "sair" in input_text.lower():
            break
        
        messages_array.append({"role": "user", "content": input_text})

        # 4. GERAÇÃO DE TEXTO (OpenAI)
        print("IA pensando...")
        response = client.chat.completions.create(
            model=azure_oai_deployment,
            messages=messages_array
        )
        
        generated_text = response.choices[0].message.content
        messages_array.append({"role": "assistant", "content": generated_text})

        # 5. SÍNTESE DE VOZ (IA falando)
        print(f"IA: {generated_text}")
        falar_texto(generated_text)

if __name__ == '__main__': 
    main()