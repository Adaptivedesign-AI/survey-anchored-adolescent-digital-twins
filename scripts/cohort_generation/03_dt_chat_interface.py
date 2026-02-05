"""
Digital Twin Chat Interface
Defines 3 DT types and provides interactive chat interface
- Survey+Memory DT: Full personalization with enriched memories
- Survey DT: Only baseline survey profile
- Base DT: Minimal generic persona
"""

from pathlib import Path
import json
from typing import Optional, List, Dict
from google import genai
from google.genai import types
import yaml


class DigitalTwinChat:
    """Interactive chat interface for Digital Twins"""
    
    # Available DT types (using paper terminology)
    DT_TYPES = {
        'survey_memory_dt': 'Survey+Memory DT',
        'survey_dt': 'Survey DT',
        'base_dt': 'Base DT'
    }
    
    # Available models
    AVAILABLE_MODELS = [
        'gemini-2.0-flash',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite'
    ]
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the chat interface"""
        self.project_root = Path(__file__).parent.parent.parent
        
        if config_path is None:
            config_path = self.project_root / "config.yaml"
        
        self.config = self._load_config(config_path)
        self._setup_paths()
        self._load_data()
        self._setup_client()
        
        # Chat state
        self.current_student_id = None
        self.current_dt_type = None
        self.current_model = None
        self.chat_history = []
    
    def _load_config(self, config_path: Path) -> dict:
        """Load configuration"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_paths(self):
        """Setup all necessary paths"""
        self.prompts_dir = self.project_root / "prompts" / "digital_twin"
        self.data_dir = self.project_root / "data" / "processed" / "cohort"
    
    def _load_data(self):
        """Load all necessary data files"""
        print("\nLoading data...")
        
        # 1. Shared base prompt (used by all DTs)
        shared_prompt_file = self.prompts_dir / "shared_base_prompt.txt"
        with open(shared_prompt_file, 'r', encoding='utf-8') as f:
            self.shared_prompt = f.read()
        print(f"✓ Loaded shared base prompt")
        
        # 2. Baseline profiles (for Survey DT and Survey+Memory DT)
        baseline_file = self.data_dir / "baseline_profiles_1000.json"
        with open(baseline_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.baseline_profiles = {
                p['student_id']: p['baseline_prompt'] 
                for p in data['profiles']
            }
        print(f"✓ Loaded {len(self.baseline_profiles)} baseline profiles")
        
        # 3. Enriched profiles (for Survey+Memory DT only)
        enriched_file = self.data_dir / "enriched_prompts_1000.json"
        with open(enriched_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.enriched_profiles = {
                e['student_id']: e 
                for e in data.get('enrichments', [])
            }
        print(f"✓ Loaded {len(self.enriched_profiles)} enriched profiles")
    
    def _setup_client(self):
        """Initialize Gemini client"""
        api_key = self.config['gemini_api_key']
        self.client = genai.Client(api_key=api_key)
    
    def _build_system_prompt(self, student_id: int, dt_type: str) -> str:
        """
        Build system prompt based on DT type
        
        DT Types:
        - survey_memory_dt: shared + baseline + enriched memories + conversations
        - survey_dt: shared + baseline only
        - base_dt: shared only (minimal generic persona)
        """
        if dt_type not in self.DT_TYPES:
            raise ValueError(f"Unknown DT type: {dt_type}")
        
        prompt_parts = []
        
        # All DTs start with shared base prompt
        prompt_parts.append(self.shared_prompt)
        
        # Survey DT and Survey+Memory DT include baseline profile
        if dt_type in ['survey_dt', 'survey_memory_dt']:
            baseline = self.baseline_profiles.get(student_id)
            if baseline:
                prompt_parts.append(f"\n\n=== Your Personal Profile ===\n{baseline}")
            else:
                print(f"⚠️  Warning: No baseline profile found for student {student_id}")
        
        # Survey+Memory DT additionally includes enriched memories
        if dt_type == 'survey_memory_dt':
            enriched = self.enriched_profiles.get(student_id)
            
            if enriched:
                # Add enriched domain narratives
                prompt_parts.append(
                    "\n\n=== Your Experiences and How You Respond ===\n"
                    "Based on your life experiences, here's how you typically think and respond:\n"
                )
                
                for domain in enriched.get('enriched_domains', []):
                    domain_name = domain.get('domain', 'Unknown')
                    narrative = domain.get('overall_domain_narrative', '')
                    prompt_parts.append(f"\n**{domain_name}**: {narrative}")
                
                # Add daily conversation examples
                daily_convos = enriched.get('daily_conversations', [])
                if daily_convos:
                    prompt_parts.append(
                        "\n\n=== Examples of Your Daily Conversations ===\n"
                    )
                    
                    for conv in daily_convos:
                        setting = conv.get('setting', '')
                        prompt_parts.append(f"\n**Setting**: {setting}")
                        
                        for dialogue in conv.get('dialogue', []):
                            speaker = dialogue.get('speaker', '')
                            text = dialogue.get('text', '')
                            prompt_parts.append(f"{speaker}: {text}")
            else:
                print(f"⚠️  Warning: No enriched profile found for student {student_id}")
        
        # Add behavioral rules
        prompt_parts.append(
            "\n\n=== Important Rules ===\n"
            "- You ARE this person. Answer as yourself, not as an AI.\n"
            "- Never mention that you are an AI, language model, or assistant.\n"
            "- Respond naturally based on your personality and experiences.\n"
            "- Be authentic and consistent with your profile."
        )
        
        return "\n".join(prompt_parts)
    
    def start_session(
        self, 
        student_id: int, 
        dt_type: str, 
        model: str = 'gemini-2.5-flash'
    ):
        """
        Start a new chat session with specified DT
        
        Args:
            student_id: Student ID (1-1000)
            dt_type: Type of DT ('survey_memory_dt', 'survey_dt', 'base_dt')
            model: Gemini model to use
        """
        # Validate inputs
        if student_id < 1 or student_id > 1000:
            raise ValueError(f"Student ID must be between 1 and 1000, got {student_id}")
        
        if dt_type not in self.DT_TYPES:
            raise ValueError(f"Invalid DT type. Choose from: {list(self.DT_TYPES.keys())}")
        
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model. Choose from: {self.AVAILABLE_MODELS}")
        
        # Set session parameters
        self.current_student_id = student_id
        self.current_dt_type = dt_type
        self.current_model = model
        self.chat_history = []
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt(student_id, dt_type)
        
        print("\n" + "="*60)
        print(f"Started chat session:")
        print(f"  Student ID: {student_id}")
        print(f"  DT Type: {self.DT_TYPES[dt_type]}")
        print(f"  Model: {model}")
        print("="*60 + "\n")
    
    def chat(self, user_message: str) -> str:
        """
        Send a message to the DT and get response
        Maintains chat history
        
        Args:
            user_message: User's message
            
        Returns:
            DT's response
        """
        if not self.current_student_id:
            raise RuntimeError("No active session. Call start_session() first.")
        
        # Add user message to history
        self.chat_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Build full prompt with history
        messages = [self.system_prompt]
        
        for msg in self.chat_history:
            if msg['role'] == 'user':
                messages.append(f"\nUser: {msg['content']}")
            else:
                messages.append(f"\nAssistant: {msg['content']}")
        
        full_prompt = "\n".join(messages) + "\nAssistant: "
        
        # Call Gemini API
        try:
            response = self.client.models.generate_content(
                model=self.current_model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048
                )
            )
            
            assistant_response = response.text.strip()
            
            # Add to history
            self.chat_history.append({
                'role': 'assistant',
                'content': assistant_response
            })
            
            return assistant_response
        
        except Exception as e:
            error_msg = f"Error calling Gemini API: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def get_chat_history(self) -> List[Dict]:
        """Get the full chat history"""
        return self.chat_history.copy()
    
    def clear_history(self):
        """Clear chat history while keeping the session"""
        self.chat_history = []
        print("✓ Chat history cleared")
    
    def end_session(self):
        """End the current chat session"""
        self.current_student_id = None
        self.current_dt_type = None
        self.current_model = None
        self.chat_history = []
        print("✓ Session ended")


def interactive_demo():
    """Interactive demo of the chat interface"""
    print("\n" + "="*60)
    print("DIGITAL TWIN CHAT INTERFACE")
    print("="*60)
    
    # Initialize
    dt_chat = DigitalTwinChat()
    
    # Get user input for session
    print("\nAvailable DT Types:")
    for key, name in DigitalTwinChat.DT_TYPES.items():
        print(f"  {key}: {name}")
    
    print("\nAvailable Models:")
    for model in DigitalTwinChat.AVAILABLE_MODELS:
        print(f"  {model}")
    
    # Example session
    print("\n" + "="*60)
    print("Starting example session...")
    print("="*60)
    
    # Start session with student 1, Survey+Memory DT, gemini-2.5-flash
    dt_chat.start_session(
        student_id=1,
        dt_type='survey_memory_dt',
        model='gemini-2.5-flash'
    )
    
    # Example conversation
    questions = [
        "Hi! Can you tell me a bit about yourself?",
        "What do you like to do in your free time?",
        "How are you feeling today?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        response = dt_chat.chat(question)
        print(f"DT: {response}")
    
    # Show history
    print("\n" + "="*60)
    print("CHAT HISTORY")
    print("="*60)
    history = dt_chat.get_chat_history()
    print(f"Total messages: {len(history)}")


def main():
    """Main entry point"""
    interactive_demo()


if __name__ == "__main__":
    main()