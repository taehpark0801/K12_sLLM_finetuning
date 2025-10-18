from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import get_peft_model

from typing import Optional

class BaseModelLoader:
    def __init__(self):
        print("BaseModelHandler initialized")

    def load(self):
        raise NotImplementedError("Subclasses must implement 'load'")

    def get_model(self):
        raise NotImplementedError("Subclasses must implement 'get_model'")

    def get_tokenizer(self):
        raise NotImplementedError("Subclasses must implement 'get_tokenizer'")
    
    
class LLMLoader(BaseModelLoader):
    def __init__(
            self, 
            model_name_or_path: Optional[str],
            model_config: Optional[dict],
            bnb_config: Optional[dict], 
            lora_config: Optional[dict],
            args: Optional[dict],
        ):
        
        super(LLMLoader, self).__init__()
        
        self.model_name_or_path = model_name_or_path
        self.model_config = model_config
        self.bnb_config = bnb_config
        self.lora_config = lora_config
        self.args = args
        
        self.model = None
        self.tokenizer = None
        
        # self.vocab_size = self.model.config.vocab_size
        # self.embed_dim = self.model.config.hidden_size

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        if "Phi" in self.model_name_or_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name_or_path,
                config=self.model_config,
                quantization_config = self.bnb_config,
                trust_remote_code=True
            )

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name_or_path,
                config=self.model_config,
                quantization_config = self.bnb_config,
            )
            
    def get_model(self):
        if self.model is None:
            raise ValueError("Model is not loaded. Please call load first.")
        
        model = get_peft_model(self.model, self.lora_config)
        # model = self.model
        return model

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded. Please call load first.")
        
        return self.tokenizer
    
class TLLMLoader(BaseModelLoader):
    def __init__(
            self, 
            model_name_or_path: Optional[str],
            model_config: Optional[dict],
            lora_config: Optional[dict], 
            peft_config: Optional[dict],
            args: Optional[dict],
        ):
        
        super(TLLMLoader, self).__init__()
        
        self.model_name_or_path = model_name_or_path
        self.model_config = model_config
        self.lora_config = lora_config
        self.peft_config = peft_config
        self.args = args
        
        self.model = None
        self.tokenizer = None
        
        # self.vocab_size = self.model.config.vocab_size
        # self.embed_dim = self.model.config.hidden_size

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        if "Phi" in self.model_name_or_path or "midm" in self.model_name_or_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name_or_path,
                config=self.model_config,
                quantization_config = self.lora_config,
                trust_remote_code=True
            )

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name_or_path,
                config=self.model_config,
                ignore_mismatched_sizes=True,
                # quantization_config = self.lora_config,
            )               
            
    def get_model(self):
        if self.model is None:
            raise ValueError("Model is not loaded. Please call load first.")
        
        # model = get_peft_model(self.model, self.peft_config)
        model = self.model
        return model

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded. Please call load first.")
        
        return self.tokenizer    