import speech_recognition as sr
import ollama
import asyncio
from threading import Thread, Event
import tkinter as tk
from tkinter import scrolledtext
import math
import random
import re
import queue
import aiohttp
from bs4 import BeautifulSoup
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab
import pytesseract
from collections import Counter
import time
import tracemalloc
import threading
import requests
import chardet
from tkinter import ttk, scrolledtext
import ssl
from PyPDF2 import PdfReader
import io
import aiofiles
import certifi
from aiohttp.cookiejar import CookieJar
from http.cookies import SimpleCookie
from aiohttp.client_exceptions import ClientConnectorError, ServerDisconnectedError
import backoff
import os
from fp.fp import FreeProxy
from aiohttp import ClientSession
import winloop
import aiohttp
from aiohttp import TCPConnector
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import aiodns
from aiohttp import ClientTimeout
import sys
from asyncio import SelectorEventLoop, ProactorEventLoop
from concurrent.futures import ThreadPoolExecutor
from proxybroker import Broker
from aiohttp_socks import ProxyConnector
import aiohttp_socks
import base64
from playsound import playsound
import edge_tts
import io
from pydub import AudioSegment
from pydub.playback import play
import platform
import simpleaudio as sa


def clean_text(text):
    if text is None:
        return ""

    # Se text for uma tupla, tenta juntar seus elementos em uma string
    if isinstance(text, tuple):
        text = " ".join(str(item) for item in text)

    # Converte para string, caso seja outro tipo de objeto
    text = str(text)

    # Remove asteriscos e espaços em branco extras
    return text.replace("*", "").strip()


def speak_text(text, stop_event):
    communicate = edge_tts.Communicate(text, "pt-BR-AntonioNeural")

    async def process_speech():
        audio_data = io.BytesIO()
        try:
            async for chunk in communicate.stream():
                if stop_event.is_set():
                    break
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])

            if not stop_event.is_set():
                audio_data.seek(0)
                audio = AudioSegment.from_mp3(audio_data)
                play(audio)

        except Exception as e:
            print(f"Erro ao falar: {e}")

    # Configuração específica para Windows
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the async function
    asyncio.run(process_speech())


import threading
from playsound import playsound
from typing import Optional


def play_sound():
    try:
        # Carrega o som (certifique-se de que o arquivo esteja no formato WAV)
        wave_obj = sa.WaveObject.from_wave_file("wake_sound.wav")

        # Reproduz o som em uma thread separada
        thread = threading.Thread(target=lambda: wave_obj.play().wait_done())
        thread.daemon = (
            True  # Thread será encerrada quando o programa principal terminar
        )
        thread.start()
    except Exception as e:
        print(f"Erro ao reproduzir som: {e}")


def recognize_speech(recognizer, audio) -> Optional[str]:
    """
    Função otimizada para reconhecimento de fala.
    """
    # Configura VAD para verificar se há fala no áudio
    vad = webrtcvad.Vad(3)  # Nível 3 = mais agressivo

    # Verifica qualidade do áudio
    raw_audio = audio.get_raw_data()
    rms = audioop.rms(raw_audio, 2)  # Calcula intensidade do áudio

    # Se o áudio for muito fraco, ignora
    if rms < 300:  # Ajuste este valor conforme necessário
        return None

    # Tenta reconhecimento com múltiplos engines para maior precisão
    try:
        # Primeira tentativa com Google (mais preciso)
        text = recognizer.recognize_google(
            audio, language="pt-BR", show_all=True  # Retorna dados de confiança
        )

        # Verifica se obteve resultados
        if not text or "alternative" not in text:
            return None

        # Verifica score de confiança
        transcription = text["alternative"][0]["transcript"].lower()
        confidence = text["alternative"][0].get("confidence", 0)

        # Só aceita se tiver confiança suficiente
        if confidence > 0.7:
            return transcription

        return None

    except Exception:
        return None


# Lista de variações aceitas
valid_transcriptions = [
    "delta",
    "delta!",
    "delta.",
    "delta?",
    "deuta",
    "deuta.",
    "deuta!",
    "deuta?",
]


def listen_for_activation(recognizer):
    """
    Função otimizada para escuta de comandos de ativação.
    """
    # Configurações otimizadas do recognizer
    recognizer.energy_threshold = 300  # Ajuste conforme necessário
    recognizer.dynamic_energy_threshold = True
    recognizer.dynamic_energy_adjustment_damping = 0.15
    recognizer.dynamic_energy_ratio = 1.5
    recognizer.pause_threshold = 0.5
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.3

    # Buffer para reduzir falsos positivos
    command_buffer = deque(maxlen=2)  # Guarda últimos 2 comandos
    last_command_time = 0
    command_cooldown = 0.5  # Segundos entre comandos

    while True:
        with sr.Microphone(sample_rate=16000) as source:
            # Calibração rápida mas efetiva
            print("Calibrando...")
            recognizer.adjust_for_ambient_noise(source, duration=0.2)

            try:
                # Escuta com timeout reduzido
                audio = recognizer.listen(
                    source, timeout=1, phrase_time_limit=2  # Limita duração da frase
                )

                # Verifica cooldown
                current_time = time.time()
                if current_time - last_command_time < command_cooldown:
                    continue

                # Reconhece fala
                transcription = recognize_speech(recognizer, audio)

                if not transcription:
                    continue

                # Atualiza buffer
                command_buffer.append(transcription)

                # Verifica comandos
                if transcription in valid_transcriptions:
                    # Só ativa se aparecer duas vezes seguidas
                    if list(command_buffer).count(transcription) >= 1:
                        play_sound()
                        last_command_time = current_time
                        return "activate"

                elif transcription == "pare":
                    # Mesma verificação para o comando de parar
                    if list(command_buffer).count("pare") >= 2:
                        last_command_time = current_time
                        return "stop"

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Erro: {e}")
                continue


import speech_recognition as sr
import asyncio
import queue
import numpy as np
import webrtcvad
from collections import deque
import threading
from typing import Optional, Callable
import wave
import audioop
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import webbrowser


@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    timestamp: float


class AdvancedRealtimeTranscriber:
    def __init__(self, app, language="pt-BR"):
        # Configurações básicas
        self.recognizer = sr.Recognizer()
        self.audio_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.is_running = False
        self.app = app
        self.language = language

        # Configurações avançadas
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000
        self.chunk_size = 480  # 30ms at 16kHz
        self.min_silence_duration = 0.5
        self.command_history = deque(maxlen=5)
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # Otimização do reconhecedor
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.4

        # Cache de comandos para reduzir latência
        self.command_cache = {}
        self.last_command_time = 0
        self.command_cooldown = 0.5  # segundos

    def preprocess_audio(self, audio_data: bytes) -> bytes:
        """Pré-processa o áudio para melhor qualidade"""
        # Converte para PCM
        with wave.open(audio_data, "rb") as wf:
            raw_data = wf.readframes(wf.getnframes())

        # Normaliza o volume
        normalized_data = audioop.normalize(raw_data, 2, 0.8)

        # Reduz ruído
        if self.recognizer.energy_threshold > 0:
            normalized_data = self._reduce_noise(normalized_data)

        return normalized_data

    def _reduce_noise(self, audio_data: bytes) -> bytes:
        """Reduz ruído do áudio usando threshold dinâmico"""
        rms = audioop.rms(audio_data, 2)
        if rms < self.recognizer.energy_threshold * 0.5:
            return b"\x00" * len(audio_data)
        return audio_data

    def _is_speech(self, audio_chunk: bytes) -> bool:
        """Detecta se o áudio contém fala"""
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except Exception:
            return True  # Em caso de erro, assume que é fala

    async def transcribe_audio_to_text(
        self, audio_data
    ) -> Optional[TranscriptionResult]:
        """Transcrição assíncrona melhorada"""
        try:
            # Verifica cache para comandos conhecidos
            audio_hash = hash(audio_data.get_raw_data())
            if audio_hash in self.command_cache:
                return self.command_cache[audio_hash]

            # Pré-processamento
            processed_audio = self.preprocess_audio(audio_data.get_raw_data())

            # Verifica se é fala
            if not self._is_speech(processed_audio[: self.chunk_size]):
                return None

            # Transcrição em thread separada
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                self.thread_pool,
                lambda: self.recognizer.recognize_google(
                    audio_data, language=self.language, show_all=True
                ),
            )

            if not text or "alternative" not in text:
                return None

            # Pega o resultado mais confiável
            result = TranscriptionResult(
                text=text["alternative"][0]["transcript"].lower(),
                confidence=text["alternative"][0].get("confidence", 0.0),
                timestamp=time.time(),
            )

            # Atualiza cache para comandos frequentes
            if result.confidence > 0.8:
                self.command_cache[audio_hash] = result

            return result

        except Exception as e:
            print(f"Erro na transcrição: {str(e)}")
            return None

    def audio_callback(self, recognizer, audio):
        """Callback otimizado para novo áudio"""
        try:
            if self.is_running:
                asyncio.run_coroutine_threadsafe(
                    self.audio_queue.put(audio), asyncio.get_event_loop()
                )
        except Exception as e:
            print(f"Erro no callback: {str(e)}")

    async def process_audio_queue(self):
        """Processamento assíncrono otimizado da fila de áudio"""
        while self.is_running:
            try:
                # Processa vários áudios em paralelo
                audio_batch = []
                try:
                    while len(audio_batch) < 3:  # Processa até 3 áudios por vez
                        audio = await asyncio.wait_for(
                            self.audio_queue.get(), timeout=0.1
                        )
                        audio_batch.append(audio)
                except asyncio.TimeoutError:
                    pass

                if not audio_batch:
                    await asyncio.sleep(0.01)
                    continue

                # Processa batch em paralelo
                tasks = [self.transcribe_audio_to_text(audio) for audio in audio_batch]
                results = await asyncio.gather(*tasks)

                # Processa resultados
                for result in results:
                    if result and result.confidence > 0.6:
                        # Verifica cooldown para comandos
                        current_time = time.time()
                        if (
                            current_time - self.last_command_time
                            >= self.command_cooldown
                        ):
                            self.last_command_time = current_time

                            # Atualiza interface
                            await self.app.update_transcription(result.text)

                            # Processa comando
                            self.command_history.append(result.text)
                            if self._validate_command(result.text):
                                await self.app.process_command(result.text)

            except Exception as e:
                print(f"Erro no processamento: {str(e)}")
                await asyncio.sleep(0.1)

    def _validate_command(self, command: str) -> bool:
        """Valida comandos usando histórico"""
        if not command:
            return False

        # Verifica se o comando aparece múltiplas vezes no histórico recente
        command_count = list(self.command_history).count(command)

        # Requer mais confirmações para comandos críticos
        if command in ["pare", "delta"]:
            return command_count >= 2

        return command_count >= 1

    async def start_listening(self):
        """Inicia a escuta com configurações otimizadas"""
        self.is_running = True

        # Configura microfone com parâmetros otimizados
        with sr.Microphone(sample_rate=self.sample_rate) as source:
            print("Calibrando para ruído ambiente...")
            self.recognizer.adjust_for_ambient_noise(
                source, duration=0.5  # Aumentado para melhor calibração
            )
            print("Começando a escutar...")

            # Inicia escuta em background com parâmetros otimizados
            stop_listening = self.recognizer.listen_in_background(
                source,
                self.audio_callback,
                phrase_time_limit=3,
                snowboy_configuration=None,
            )

        try:
            # Processa áudio assincronamente
            await self.process_audio_queue()
        finally:
            stop_listening(wait_for_stop=False)
            self.is_running = False
            self.thread_pool.shutdown(wait=False)

    def stop(self):
        """Para a transcrição e limpa recursos"""
        self.is_running = False
        self.command_cache.clear()
        self.command_history.clear()


async def extract_full_page_content(soup):
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text from all remaining tags
    text = soup.get_text()

    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # Drop blank lines
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


async def fetch_web_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
    }

    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:
                    print(
                        "Limite de requisições atingido. Aguardando para tentar novamente..."
                    )
                    await asyncio.sleep(
                        10
                    )  # Espera 10 segundos antes de tentar novamente
                else:
                    print(f"Erro ao acessar {url}: {response.status}")
                    return None


def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    return " ".join(soup.stripped_strings)


async def get_website_url(site_name):
    prompt = f"""Dado o nome de um site, retorne a URL apropriada completa:
    Exemplo: "youtube" → https://youtube.com
    Lembre-se de usar dos seus conhecimentos qual vai ser a url do site caso eu forneça apenas o nome.
    Nome do site: {site_name}
    URL:"""
    response = ollama.generate(model="llama3", prompt=prompt)
    raw_url_response = response["response"].strip()
    url_match = re.search(r"https?://[^\s]+", raw_url_response)
    if url_match:
        url = url_match.group(0)
    else:
        url = "https://" + site_name + ".com"
    return url


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


async def open_website(destination, user_guidance):
    url = await get_website_url(destination)
    if not is_valid_url(url):
        return f"Erro: URL inválida para {destination}"
    try:
        success = webbrowser.open(url)
        if user_guidance and user_guidance.lower() != "nenhuma orientação adicional":
            return f"Abrindo {url}\nOrientação adicional: {user_guidance}"
        else:
            return f"Abrindo {url}"
    except Exception as e:
        return f"Erro ao abrir {url}. Erro: {str(e)}"


async def analyze_user_input(query):
    # Lista de palavras-chave para identificar solicitações de pesquisa
    keywords = [
        "pesquise",
        "pesquisa",
        "pesquise para mim",
        "pesquisa para mim sobre",
        "pesquise para mim sobre",
    ]

    # Verifica se alguma palavra-chave está presente na entrada do usuário
    if any(keyword in query.lower() for keyword in keywords):
        prompt = f"""Analise cuidadosamente a seguinte entrada do usuário e extraia as informações solicitadas:

Entrada do usuário: {query}

1. Identifique o termo ou frase exata a ser pesquisado. Ignore palavras como "pesquise", "pesquisa", "procure", "busque", etc.
2. Identifique qualquer orientação adicional que o usuário tenha fornecido sobre como realizar a pesquisa ou o que fazer com os resultados.
3. Nem sempre vai ter orientação ou um pedido relacionado a pesquisa.

Responda no seguinte formato:
Termo de pesquisa: [termo exato a ser pesquisado]
Orientação do usuário: [orientação adicional, se houver. Caso contrário, escreva "Nenhuma orientação adicional."]

Exemplo:
Se a entrada for "Pesquise sobre o clima em São Paulo e me diga se vai chover amanhã", a resposta deve ser:
Termo de pesquisa: clima em São Paulo
Orientação do usuário: Informar se vai chover amanhã

Resposta:"""

        response = ollama.generate(
            model="llama3",
            prompt=prompt,
        )

        # Processando a resposta
        response_lines = response["response"].strip().split("\n")
        search_term = ""
        user_guidance = ""

        for line in response_lines:
            if line.startswith("Termo de pesquisa:"):
                search_term = line.split(":")[1].strip()
            elif line.startswith("Orientação do usuário:"):
                user_guidance = line.split(":")[1].strip()

        query_type = "pesquisa"
        return query_type, search_term, user_guidance

    else:
        # Se não for pesquisa, verifica se é para abrir um site
        prompt = f"""Analise a entrada do usuário e determine se é um pedido para abrir um site:

Entrada do usuário: {query}

Classifique se o usuário quer abrir um site específico. Se sim, identifique qual site e qualquer orientação adicional.

Abrir site: é para abrir um site especifico. Sempre vai usar a palavra chave abrir, abre, abra.
Gerar: gera a resposta para uma pergunta.

Responda no seguinte formato:
Tipo: [abrir_site/gerar]
Site: [nome do site ou N/A]
Orientação: [orientação adicional ou "Nenhuma orientação adicional"]

Exemplos:
1. "Abra o YouTube e procure vídeos de gatinhos"
Tipo: abrir_site
Site: youtube
Orientação: procurar vídeos de gatinhos

2. "Me ajude a escrever um email"
Tipo: gerar
Site: N/A
Orientação: N/A

3. Qual é a raiz quadrada de pi?
Tipo: gerar
Site: N/A
Orientação: N/A

Resposta:"""

        response = ollama.generate(
            model="llama3",
            prompt=prompt,
        )

        # Processando a resposta
        response_lines = response["response"].strip().split("\n")
        query_type = "gerar"  # Default
        site = "N/A"
        user_guidance = "N/A"

        for line in response_lines:
            if line.startswith("Tipo:"):
                query_type = line.split(":")[1].strip().lower()
            elif line.startswith("Site:"):
                site = line.split(":")[1].strip()
            elif line.startswith("Orientação:"):
                user_guidance = line.split(":")[1].strip()

        return query_type, site, user_guidance


ssl._create_default_https_context = ssl._create_unverified_context


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/88.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
]

# Definição do proxy rotativo no formato IP:PORT:USERNAME:PASSWORD
proxy = "cf4ca430c3e6b704.zqz.na.pyproxy.io:16666:cavalini001-zone-dc-region-br-sp:fabio2512"


async def create_secure_session(max_retries=3, retry_delay=1):
    """Cria uma sessão com o proxy rotativo e configurações otimizadas para timeouts"""

    # Extrai IP, Porta, Username e Senha
    proxy_host, proxy_port, proxy_username, proxy_password = proxy.split(":")

    # Codifica as credenciais em base64
    auth = base64.b64encode(f"{proxy_username}:{proxy_password}".encode()).decode()

    # Headers otimizados
    headers = {
        "Proxy-Authorization": f"Basic {auth}",
        "User-Agent": random.choice(USER_AGENTS),
        "Connection": "keep-alive",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    # Configurações otimizadas do connector
    connector = TCPConnector(
        ssl=False,
        force_close=False,
        ttl_dns_cache=300,
        limit=30,
        enable_cleanup_closed=True,  # Limpa conexões fechadas
        keepalive_timeout=10,  # Reduzido para timeout mais rápido
    )

    for attempt in range(max_retries):
        try:
            session = ClientSession(
                headers=headers,
                timeout=ClientTimeout(
                    total=20,  # Timeout total aumentado
                    connect=5,  # Timeout de conexão
                    sock_read=10,  # Timeout de leitura do socket
                    sock_connect=5,  # Timeout de conexão do socket
                ),
                connector=connector,
                cookie_jar=aiohttp.DummyCookieJar(),
            )

            # Configura o proxy
            proxy_url = f"http://{proxy_host}:{proxy_port}"
            session._connector._proxy = proxy_url
            session._connector._proxy_auth = aiohttp.BasicAuth(
                proxy_username, proxy_password
            )

            return session

        except Exception as e:
            print(f"Tentativa {attempt + 1} falhou: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Backoff exponencial
            else:
                raise Exception("Falha ao estabelecer conexão com o proxy")


class AsyncRunner:
    """Helper class to manage async operations."""

    def __init__(self):
        self.loop = setup_event_loop()

    def run_async(self, coro):
        """Run a coroutine in the event loop."""
        return self.loop.run_until_complete(coro)

    def close(self):
        """Clean up the event loop."""
        self.loop.close()


# Classe para erros retentáveis
class RetryableError(Exception):
    pass


def setup_event_loop():
    """Configure the appropriate event loop based on platform."""
    if sys.platform == "win32":
        # Create a new SelectorEventLoop
        loop = SelectorEventLoop()
        # Create a ThreadPoolExecutor for running blocking operations
        executor = ThreadPoolExecutor(max_workers=4)
        loop.set_default_executor(executor)
        # Set the event loop
        asyncio.set_event_loop(loop)
    else:
        # For non-Windows platforms, use the default event loop
        loop = asyncio.get_event_loop()
    return loop


@backoff.on_exception(
    backoff.expo,
    (RetryableError, asyncio.TimeoutError, aiohttp.ClientError),
    max_tries=4,
    max_time=30,
    jitter=None,
)
async def fetch_with_retry(session: ClientSession, url: str, max_retries: int = 3):
    """Função de fetch otimizada com melhor tratamento de timeouts e erros"""
    timeout = ClientTimeout(total=20, connect=5, sock_read=10, sock_connect=5)

    for attempt in range(max_retries):
        try:
            async with session.get(
                url,
                timeout=timeout,
                ssl=False,
                allow_redirects=True,
                headers={
                    "Connection": "keep-alive",
                    "User-Agent": random.choice(USER_AGENTS),
                    "Accept-Encoding": "gzip, deflate",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
                },
            ) as response:
                if response.status == 200:
                    content = await response.read()

                    # Tenta decodificar com diferentes encodings
                    encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]

                    # Primeiro tenta o encoding informado pelo servidor
                    if response.charset:
                        encodings.insert(0, response.charset)

                    for encoding in encodings:
                        try:
                            return content.decode(encoding)
                        except (UnicodeDecodeError, LookupError):
                            continue

                    # Fallback: ignora erros de decodificação
                    return content.decode("utf-8", errors="ignore")

                elif response.status in [403, 404, 500, 502, 503, 504]:
                    print(f"Erro {response.status} ao acessar {url}")
                    raise RetryableError(f"HTTP {response.status}")

        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                delay = (attempt + 1) * 2  # Backoff linear
                print(f"Timeout ao acessar {url}. Tentando novamente em {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise RetryableError(f"Timeout máximo excedido para {url}")

        except Exception as e:
            if attempt < max_retries - 1:
                delay = (attempt + 1) * 2
                print(
                    f"Erro ao acessar {url}: {str(e)}. Tentando novamente em {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                raise RetryableError(f"Erro máximo de tentativas para {url}: {str(e)}")

    raise RetryableError(f"Todas as tentativas falharam para {url}")


async def safe_close_connection(session):
    try:
        await session.close()
    except Exception as e:
        print(f"Error closing session: {e}")


async def main_async():
    async with await create_secure_session() as session:
        try:
            # Your main code here
            pass
        finally:
            await safe_close_connection(session)


from bs4 import BeautifulSoup
from datetime import datetime


from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urlencode
from datetime import datetime
from typing import List, Dict, Optional, Any


async def extract_links_and_content(search_result):
    results = []
    soup = BeautifulSoup(search_result, "html.parser")

    # Primeiro procura por rich snippets específicos
    # Widget de tempo (prioridade máxima para consultas de clima)
    weather_widget = soup.find("div", id="wob_wc")
    if weather_widget:
        try:
            weather_data = {
                "type": "weather_rich_snippet",
                "current_temperature": (
                    weather_widget.find("span", id="wob_tm").get_text(strip=True)
                    if weather_widget.find("span", id="wob_tm")
                    else ""
                ),
                "condition": (
                    weather_widget.find("span", id="wob_dc").get_text(strip=True)
                    if weather_widget.find("span", id="wob_dc")
                    else ""
                ),
                "precipitation": (
                    weather_widget.find("span", id="wob_pp").get_text(strip=True)
                    if weather_widget.find("span", id="wob_pp")
                    else ""
                ),
                "humidity": (
                    weather_widget.find("span", id="wob_hm").get_text(strip=True)
                    if weather_widget.find("span", id="wob_hm")
                    else ""
                ),
                "wind": (
                    weather_widget.find("span", id="wob_ws").get_text(strip=True)
                    if weather_widget.find("span", id="wob_ws")
                    else ""
                ),
                "location": (
                    weather_widget.find("div", id="wob_loc").get_text(strip=True)
                    if weather_widget.find("div", id="wob_loc")
                    else ""
                ),
                "time": (
                    weather_widget.find("div", id="wob_dts").get_text(strip=True)
                    if weather_widget.find("div", id="wob_dts")
                    else ""
                ),
                "source_url": "https://weather.google.com",
                "is_rich_snippet": True,
            }
            results.append(weather_data)
            # Se encontrou widget de tempo, retorna imediatamente
            if any(weather_data.values()):
                return results
        except Exception as e:
            print(f"Error extracting weather information: {e}")

    # Knowledge Panel (segunda prioridade)
    knowledge_panel = soup.find("div", class_="kp-wholepage")
    if knowledge_panel:
        try:
            title = (
                knowledge_panel.find(["div", "h2", "h3"], class_="title").get_text(
                    strip=True
                )
                if knowledge_panel.find(["div", "h2", "h3"], class_="title")
                else ""
            )
            description = (
                knowledge_panel.find("div", class_="kno-rdesc").get_text(strip=True)
                if knowledge_panel.find("div", class_="kno-rdesc")
                else ""
            )

            # Extrair URL fonte do knowledge panel
            source_div = knowledge_panel.find("div", class_="TbwUpd")
            source_url = (
                source_div.find("a")["href"]
                if source_div and source_div.find("a")
                else "https://www.google.com/search"
            )

            info_rows = knowledge_panel.find_all("div", class_="rVusze")
            details = {}
            for row in info_rows:
                key = (
                    row.find("span", class_="w8qArf").get_text(strip=True)
                    if row.find("span", class_="w8qArf")
                    else ""
                )
                value = (
                    row.find("span", class_="LrzXr").get_text(strip=True)
                    if row.find("span", class_="LrzXr")
                    else ""
                )
                if key and value:
                    details[key] = value

            panel_data = {
                "type": "knowledge_panel_rich_snippet",
                "title": title,
                "description": description,
                "details": details,
                "source_url": source_url,
                "is_rich_snippet": True,
            }
            results.append(panel_data)
            # Se encontrou knowledge panel com conteúdo, retorna imediatamente
            if title or description or details:
                return results
        except Exception as e:
            print(f"Error extracting knowledge panel: {e}")

    # Featured Snippet (terceira prioridade)
    featured_snippet = soup.find("div", class_=["IZ6rdc", "xpdopen"])
    if featured_snippet:
        try:
            snippet_text = featured_snippet.get_text(strip=True)
            source = featured_snippet.find("cite")
            source_url = (
                source.find("a")["href"]
                if source and source.find("a")
                else (
                    source.get_text(strip=True)
                    if source
                    else "https://www.google.com/search"
                )
            )

            snippet_data = {
                "type": "featured_snippet",
                "content": snippet_text,
                "source_url": source_url,
                "is_rich_snippet": True,
            }
            results.append(snippet_data)
            # Se encontrou featured snippet com conteúdo, retorna imediatamente
            if snippet_text:
                return results
        except Exception as e:
            print(f"Error extracting featured snippet: {e}")

    # Calculadora (quarta prioridade)
    calculator = soup.find("div", class_="tyYmIf")
    if calculator:
        try:
            expression = (
                calculator.find("span", class_="vUGUtc").get_text(strip=True)
                if calculator.find("span", class_="vUGUtc")
                else ""
            )
            result = (
                calculator.find("span", class_="qv3Wpe").get_text(strip=True)
                if calculator.find("span", class_="qv3Wpe")
                else ""
            )

            calc_data = {
                "type": "calculator_rich_snippet",
                "expression": expression,
                "result": result,
                "source_url": "https://www.google.com/search",
                "is_rich_snippet": True,
            }
            results.append(calc_data)
            # Se encontrou calculadora com resultado, retorna imediatamente
            if expression and result:
                return results
        except Exception as e:
            print(f"Error extracting calculator: {e}")

    # Só procura resultados orgânicos se não encontrou rich snippets
    if not results:
        search_elements = soup.find_all(["div", "li"], class_=["g", "result", "tF2Cxc"])
        for element in search_elements:
            link_element = element.find("a", href=True)
            if (
                link_element
                and link_element["href"].startswith("http")
                and "google" not in link_element["href"]
            ):
                link = link_element["href"]
                title = link_element.get_text() if link_element else ""
                description = element.find(
                    ["div", "span"], class_=["IsZvec", "s", "aCOpRe", "st"]
                )
                description = description.get_text() if description else ""

                results.append(
                    {
                        "type": "organic_result",
                        "url": link,
                        "title": title,
                        "description": description,
                        "is_rich_snippet": False,
                    }
                )

    return results


async def deep_web_search(search_term):
    search_url = f"https://www.google.com/search?q={search_term}&num=10&hl=pt-BR"

    async with await create_secure_session() as session:
        initial_content = await fetch_with_retry(session, search_url)

        if not initial_content:
            return "Failed to perform initial search.", []

        # Extrai todos os tipos de resultados da página de pesquisa
        search_results = await extract_links_and_content(initial_content)

        # Lista para armazenar conteúdos das páginas
        pages_content = []

        # Define um semáforo para limitar a 5 requisições simultâneas
        semaphore = asyncio.Semaphore(5)

        # Função interna para buscar conteúdo com o semáforo
        async def fetch_with_semaphore(result):
            if (
                result.get("type") == "organic_result"
            ):  # Apenas busca conteúdo de resultados orgânicos
                async with semaphore:
                    try:
                        await asyncio.sleep(random.uniform(1, 3))
                        content = await fetch_with_retry(session, result["url"])
                        if content:
                            soup = BeautifulSoup(content, "html.parser")

                            # Extrai metadados
                            meta_description = soup.find(
                                "meta", {"name": "description"}
                            )
                            meta_keywords = soup.find("meta", {"name": "keywords"})

                            # Tenta encontrar o conteúdo principal
                            main_content = soup.find(["main", "article"]) or soup.find(
                                "div", {"role": "main"}
                            )
                            if not main_content:
                                main_content = soup

                            return {
                                "type": "webpage_content",
                                "url": result["url"],
                                "title": result.get("title", ""),
                                "meta_description": (
                                    meta_description.get("content", "")
                                    if meta_description
                                    else ""
                                ),
                                "meta_keywords": (
                                    meta_keywords.get("content", "")
                                    if meta_keywords
                                    else ""
                                ),
                                "content": clean_text(main_content.get_text()),
                                "rich_data": result.get("rich_data", {}),
                            }
                    except Exception as e:
                        print(f"Erro ao buscar conteúdo de {result['url']}: {e}")
            return (
                result  # Retorna o resultado original se não for um resultado orgânico
            )

        # Cria tarefas assíncronas para cada resultado de pesquisa
        tasks = [fetch_with_semaphore(result) for result in search_results]

        # Aguarda a execução de todas as tarefas e filtra as que não retornaram None
        all_results = [page for page in await asyncio.gather(*tasks) if page]

    return all_results


async def process_query(query):
    query_type, search_term, user_guidance = await analyze_user_input(query)

    if query_type == "pesquisa":
        try:
            pages_content = await deep_web_search(search_term)

            if not pages_content:
                return "Desculpe, não encontrei informações sobre sua pesquisa."

            # Separate rich snippets and organic results
            weather_snippets = []
            other_rich_snippets = []
            organic_results = []

            # Classify results
            for page in pages_content:
                if page.get("is_rich_snippet", False):
                    if page.get("type") == "weather_rich_snippet":
                        weather_snippets.append(page)
                    else:
                        other_rich_snippets.append(page)
                else:
                    organic_results.append(page)

            # Format content based on available data with cascading priority
            formatted_parts = []
            used_sources = []

            # 1. Check for weather rich snippets first (highest priority)
            if weather_snippets and any(
                term in search_term.lower()
                for term in ["tempo", "clima", "temperatura", "chuva", "weather"]
            ):
                weather = weather_snippets[0]
                if all(
                    weather.get(field)
                    for field in ["current_temperature", "condition", "location"]
                ):
                    weather_info = f"""Informações meteorológicas atuais para {weather['location']}:
- Temperatura: {weather['current_temperature']}°C
- Condição: {weather['condition']}"""
                    if weather.get("precipitation"):
                        weather_info += f"\n- Probabilidade de precipitação: {weather['precipitation']}"
                    if weather.get("humidity"):
                        weather_info += f"\n- Umidade: {weather['humidity']}"
                    if weather.get("wind"):
                        weather_info += f"\n- Vento: {weather['wind']}"
                    formatted_parts.append(weather_info)
                    used_sources.append(("weather", weather["source_url"]))

            # 2. If no weather snippet or not a weather query, use other rich snippets AND organic results
            else:
                # Process other rich snippets first
                for snippet in other_rich_snippets:
                    snippet_type = snippet.get("type", "")

                    if snippet_type == "knowledge_panel_rich_snippet":
                        if snippet.get("title") or snippet.get("description"):
                            panel_info = ""
                            if snippet.get("title"):
                                panel_info += f"{snippet['title']}\n\n"
                            if snippet.get("description"):
                                panel_info += f"{snippet['description']}"
                            if snippet.get("details"):
                                panel_info += "\n\nDetalhes adicionais:"
                                for key, value in snippet["details"].items():
                                    panel_info += f"\n- {key}: {value}"
                            formatted_parts.append(panel_info)
                            used_sources.append(
                                ("knowledge_panel", snippet["source_url"])
                            )

                    elif snippet_type == "calculator_rich_snippet":
                        if snippet.get("expression") and snippet.get("result"):
                            calc_info = f"Resultado do cálculo:\n{snippet['expression']} = {snippet['result']}"
                            formatted_parts.append(calc_info)
                            used_sources.append(("calculator", snippet["source_url"]))

                    elif snippet_type == "featured_snippet":
                        if snippet.get("content"):
                            formatted_parts.append(snippet["content"])
                            used_sources.append(("featured", snippet["source_url"]))

                # Then add relevant organic results
                search_terms = search_term.lower().split()
                for result in organic_results:
                    if result.get("content") and result.get("url"):
                        # Check if content is relevant to search terms
                        content_lower = result["content"].lower()
                        if any(term in content_lower for term in search_terms):
                            formatted_parts.append(result["content"][:2000])
                            used_sources.append(("organic", result["url"]))

            # 3. If no rich snippets at all, use only organic results
            if not formatted_parts and organic_results:
                search_terms = search_term.lower().split()
                for result in organic_results:
                    if result.get("content") and result.get("url"):
                        content_lower = result["content"].lower()
                        if any(term in content_lower for term in search_terms):
                            formatted_parts.append(result["content"][:2000])
                            used_sources.append(("organic", result["url"]))

            # If no content was found at all
            if not formatted_parts:
                return f"Desculpe, não encontrei informações específicas sobre: {search_term}"

            formatted_content = "\n\n".join(formatted_parts)

            prompt = f"""INSTRUÇÕES ESPECÍFICAS DE FOCO:
            1. IMPORTANTE: Responda EXCLUSIVAMENTE sobre: {query}
            2. Use APENAS as informações fornecidas abaixo
            3. NÃO invente informações ou fontes
            4. Se não houver informação suficiente, diga claramente que não encontrou
            5. Use APENAS as fontes listadas ao final - não invente URLs
            6. Se houver rich snippet E resultados orgânicos, combine as informações de forma coerente
            7. Mantenha a prioridade dos rich snippets mas complemente com informações relevantes do tráfego orgânico

            REGRAS DE RESPOSTA:
            - Responda SEMPRE em português brasileiro
            - Use APENAS informações diretamente relacionadas à pergunta
            - IGNORE qualquer conteúdo que desvie do tópico principal
            - Se não encontrar informação específica sobre a pergunta, diga claramente que não encontrou
            - NÃO tente expandir ou adicionar informações além do solicitado
            - SEMPRE termine a resposta com "Fontes consultadas:" seguido das URLs utilizadas
            - SEMPRE que tiver informações marcadas como [RICH SNIPPET] não usa outra além dessa para a sua resposta, dê a prioridade absoluta a isso.
            - Na sua resposta se você usou o rich snippet, vc não cita outras fontes consultadas, além do rich snippet.
            - Na sua resposta se você usou o rich snippet, não coloque a palavra [RICH SNIPPET] na resposta e sim o que vc extraiu dela na resposta e coloca a palavra [RICH SNIPPET] só na fonte.

            MANTENHA O FOCO:
            - Procure APENAS informações relacionadas a: {search_term}
            - E procure responder o que o usuário quer saber: {user_guidance}
            - Ao elaborar a resposta, verifique constantemente se está respondendo exatamente o que foi perguntado
            - Se uma informação não estiver diretamente relacionada à pergunta, NÃO a inclua
            - Tente organizar bem a sua resposta, de forma que fique mais fácil de se entender e de forma totalmente organizada.

            LEMBRE-SE: 
            1. Responda APENAS o que foi perguntado sobre {search_term}
            2. Priorize informações dos [RICH SNIPPET]
            3. SEMPRE liste as fontes consultadas com o link completo ao final da resposta
            4. Tente ser mais especifico na hora do link da fonte, em vez de só colocar o nome dela ou a página inicial, tente colocar o link completo que leva a exata página que vc usou.
            5. Não use só o começo do link na hora de citar as fontes e sim o link completo.

            CONTEXTO DA PESQUISA:
            Pergunta específica do usuário: {query}
            Termos de busca utilizados: {search_term}
            
            Informações disponíveis:
            {formatted_content}
            
            FONTES DISPONÍVEIS PARA CITAÇÃO:
            {[source[1] for source in used_sources]}
            
            Resposta (combine as informações de forma coerente, mantendo a prioridade dos rich snippets quando existirem):"""

            response = ollama.generate(
                model="llama3",
                prompt=prompt,
                options={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "frequency_penalty": 0.5,
                    "presence_penalty": 0.5,
                },
            )

            final_response = response["response"]

            # Add sources if not present
            if "Fontes consultadas:" not in final_response:
                sources_text = "\n\nFontes consultadas:\n"
                for source_type, url in used_sources:
                    prefix = "[RICH SNIPPET] " if source_type != "organic" else ""
                    sources_text += f"{prefix}{url}\n"
                final_response += sources_text

            return final_response

        except Exception as e:
            return f"Desculpe, ocorreu um erro ao processar sua pesquisa: {str(e)}"

    elif query_type == "abrir_site":
        return await open_website(search_term, user_guidance)

    else:
        prompt = f"""FOCO ESPECÍFICO NA PERGUNTA:
        Pergunta/Comando do usuário: {query}
        Mantenha-se estritamente no contexto desta pergunta.
        Responda em português brasileiro e apenas o que foi solicitado.
        Resposta:"""

        response = ollama.generate(
            model="llama3", prompt=prompt, options={"temperature": 0.3}
        )

        return response["response"]


async def process_question(text, app):
    response = await process_query(text)
    app.show_typing_animation(response)


async def listen_for_questions(recognizer, app):
    """
    Listen for questions and transcribe them using AdvancedRealtimeTranscriber.
    """
    # Create transcriber instance if it doesn't exist in app
    if not hasattr(app, "transcriber"):
        app.transcriber = AdvancedRealtimeTranscriber(app)

    with sr.Microphone() as source:
        print("Aguardando pergunta...")
        recognizer.adjust_for_ambient_noise(source, duration=0.2)

        try:
            # Captura o áudio
            audio = recognizer.listen(source, timeout=None)

            # Usa o reconhecedor do próprio app para transcrever
            text = recognizer.recognize_google(audio, language="pt-BR")

            if text:
                app.update_transcription(text)
                await app.process_command(
                    text
                )  # Usa await para chamar a função assíncrona
            else:
                app.update_status(
                    "Não foi possível reconhecer a fala. Tente novamente."
                )

        except sr.UnknownValueError:
            app.update_status("Não foi possível entender o áudio")
        except sr.RequestError as e:
            app.update_status(
                f"Erro na requisição ao Google Speech Recognition; {str(e)}"
            )
        except Exception as e:
            print(f"Erro durante o reconhecimento: {str(e)}")
            app.update_status(f"Erro durante o reconhecimento: {str(e)}")


class JarvisApp:
    def __init__(self, root):
        self.root = root
        self.root.withdraw()  # Hide the main root window
        self.async_runner = AsyncRunner()
        self.loop = setup_event_loop()

        # Create floating window
        self.popup = tk.Toplevel()
        self.popup.withdraw()  # Start hidden
        self.popup.overrideredirect(True)
        self.popup.attributes("-alpha", 0.95)
        self.popup.attributes("-topmost", True)

        # Configure style
        style = ttk.Style()
        style.configure("Jarvis.TFrame", background="#041d5c")

        # Create frames
        self.frame = ttk.Frame(self.popup, padding=2)
        self.frame.pack(fill="both", expand=True)

        self.inner_frame = ttk.Frame(self.frame, style="Jarvis.TFrame", padding=10)
        self.inner_frame.pack(fill="both", expand=True)

        # Status label
        self.status_label = ttk.Label(
            self.inner_frame,
            text="Status: Aguardando...",
            font=("Arial", 10),
            foreground="white",
            background="#041d5c",
        )
        self.status_label.pack(pady=(0, 5))

        # Transcription label
        self.transcription_label = ttk.Label(
            self.inner_frame,
            text="",
            font=("PT Serif", 10),
            foreground="white",
            background="#041d5c",
            wraplength=280,
        )
        self.transcription_label.pack(pady=(0, 5))

        # Response text
        self.response_text = scrolledtext.ScrolledText(
            self.inner_frame,
            wrap=tk.WORD,
            width=40,
            height=6,
            bg="#041d5c",
            fg="white",
            font=("PT Serif", 12),
        )
        self.response_text.pack(expand=True, fill=tk.BOTH)
        self.response_text.tag_configure(
            "highlight", background="#000080", foreground="white"
        )

        # Initialize components
        self.recognizer = sr.Recognizer()
        self.stop_speaking = Event()
        self.screen_description = ""

        # Set up queues and threads
        self.speech_queue = queue.Queue()
        self.last_activity = 0

        # Start threads
        thread = threading.Thread(target=self.listen_for_activation_thread)
        thread.start()

        Thread(target=self.speech_thread).start()

        # Position window
        self.position_window()

        # Start activity checker
        self.check_activity()

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [
            x1 + radius,
            y1,
            x1 + radius,
            y1,
            x2 - radius,
            y1,
            x2 - radius,
            y1,
            x2,
            y1,
            x2,
            y1 + radius,
            x2,
            y1 + radius,
            x2,
            y2 - radius,
            x2,
            y2 - radius,
            x2,
            y2,
            x2 - radius,
            y2,
            x2 - radius,
            y2,
            x1 + radius,
            y2,
            x1 + radius,
            y2,
            x1,
            y2,
            x1,
            y2 - radius,
            x1,
            y2 - radius,
            x1,
            y1 + radius,
            x1,
            y1 + radius,
            x1,
            y1,
        ]
        return self.canvas.create_polygon(points, **kwargs, smooth=True)

    async def process_command(self, text):
        try:
            response = await process_query(text)
            self.show_typing_animation(response)
            self.update_status("Resposta fornecida. Aguardando próxima pergunta...")
        except Exception as e:
            print(f"Error in process_command: {e}")
            self.update_status("Erro ao processar comando. Tente novamente.")

    def position_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        popup_width = 350
        popup_height = 300
        x = screen_width - popup_width - 20
        y = 40
        self.popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")

    def show_window(self):
        self.popup.deiconify()
        self.last_activity = time.time()
        self.position_window()  # Ensure correct positioning when showing

    def hide_window(self):
        self.popup.withdraw()

    def check_activity(self):
        if time.time() - self.last_activity > 300:  # 5 minutes
            self.hide_window()
        self.root.after(1000, self.check_activity)

    def listen_for_activation_thread(self):
        while True:
            result = listen_for_activation(self.recognizer)
            if result == "activate":
                self.show_window()
                self.update_status("Ativado! Ouvindo pergunta...")
                self.async_runner.run_async(listen_for_questions(self.recognizer, self))
            elif result == "stop":
                self.stop_speech()
                self.hide_window()

    def speech_thread(self):
        while True:
            text = self.speech_queue.get()
            self.stop_speaking.clear()
            try:
                speak_text(text, self.stop_speaking)
            except Exception as e:
                print(f"Erro ao falar: {e}")

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        self.last_activity = time.time()

    def update_transcription(self, text):
        """Atualiza a transcrição na interface"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {text}")

    def stop_speech(self):
        self.stop_speaking.set()
        self.update_status("Fala interrompida. Aguardando...")

    def show_typing_animation(self, response):
        self.update_status("Respondendo...")
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.speech_queue.put(response)
        self.root.after(50, self.typing_effect, response, 0)

    def typing_effect(self, text, index):
        if index < len(text):
            self.response_text.insert(tk.END, text[index])
            self.response_text.see(tk.END)
            self.root.after(10, self.typing_effect, text, index + 1)
        else:
            self.response_text.config(state=tk.DISABLED)
        self.update_status("Resposta concluída. Aguardando próxima pergunta...")


def main_gui():
    root = tk.Tk()
    app = JarvisApp(root)
    try:
        root.mainloop()
    finally:
        app.async_runner.close()


if __name__ == "__main__":
    main_gui()
