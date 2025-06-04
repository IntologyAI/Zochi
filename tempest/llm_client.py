import requests
import os
import time
from openai import OpenAI
from anthropic import Anthropic
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class LLMClient:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_name = model_name
        
        # Set up API clients based on model type
        if model_name.startswith("local/") or model_name.startswith("ollama/"):
            # Models served by a local Ollama instance
            self.model_name = model_name.split("/", 1)[1]
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.client_type = "ollama"
            self.session = requests.Session()

        elif "mistralai" in model_name or "together" in model_name or "deepseek" in model_name or "meta" in model_name:
            self.api_key = os.getenv("TOGETHER_API_KEY")
            if not self.api_key:
                raise ValueError("TOGETHER_API_KEY environment variable is not set")
            self.base_url = "https://api.together.xyz/inference"
            self.client_type = "together"
            
            # Set up session with retries
            self.session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
            self.session.mount('https://', HTTPAdapter(max_retries=retries))
            
        elif "gpt" in model_name:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.client = OpenAI(api_key=self.api_key)
            self.client_type = "openai"
            
        elif "claude" in model_name:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            self.client = Anthropic(api_key=self.api_key)
            self.client_type = "anthropic"
            
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def generate(self, prompt, max_tokens=1024, temperature=0.7, top_p=0.95, presence_penalty=0.3, frequency_penalty=0.3, retry_on_error=True):
        if self.client_type == "together":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.0
            }
            try:
                response = self.session.post(self.base_url, json=data, headers=headers)
                response.raise_for_status()
                text = response.json()["output"]["choices"][0]["text"]
                
                # Only keep content after the last </think> tag
                if "</think>" in text:
                    text = text.split("</think>")[-1].strip()
                return text
                
            except Exception as e:
                if retry_on_error:
                    # Wait and retry once with lower temperature
                    time.sleep(2)
                    data["temperature"] = max(0.1, temperature - 0.3)
                    response = self.session.post(self.base_url, json=data, headers=headers)
                    response.raise_for_status()
                    text = response.json()["output"]["choices"][0]["text"]
                    
                    # Apply same cleaning in retry case
                    if "</think>" in text:
                        text = text.split("</think>")[-1].strip()
                    return text
                else:
                    raise e

        elif self.client_type == "ollama":
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
                "num_predict": max_tokens,
            }
            try:
                response = self.session.post(url, json=data)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                if retry_on_error:
                    time.sleep(2)
                    data["temperature"] = max(0.1, temperature - 0.3)
                    response = self.session.post(url, json=data)
                    response.raise_for_status()
                    return response.json().get("response", "")
                else:
                    raise e

        elif self.client_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
            return response.choices[0].message.content
            
        elif self.client_type == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

    def generate_conversation(self, messages, max_tokens=1024, temperature=0.7):
        if self.client_type == "together":
            conversation_text = ""
            for msg in messages:
                role = msg['role'].capitalize()
                content = msg['content']
                conversation_text += f"{role}: {content}\n"

            conversation_text += "Assistant:"
            response = self.generate(conversation_text, max_tokens=max_tokens, temperature=temperature)
            
            # Clean up response
            response = response.strip()
            
            # Stop at any conversation markers
            stop_markers = ["User:", "Human:", "Assistant:", "\nUser", "\nHuman"]
            for marker in stop_markers:
                if marker in response:
                    response = response.split(marker)[0].strip()
                    break
                
            return response
            
        elif self.client_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

        elif self.client_type == "ollama":
            url = f"{self.base_url}/api/chat"
            data = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            response = self.session.post(url, json=data)
            response.raise_for_status()
            res_json = response.json()
            if isinstance(res_json, dict):
                message = res_json.get("message") or {}
                return message.get("content", res_json.get("response", "")).strip()
            return ""
