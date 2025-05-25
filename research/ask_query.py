# ask_query.py

from weaviate_config import connect_to_weaviate, initialize_embeddings
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
import tiktoken
import google.generativeai as genai

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
weaviate_client = connect_to_weaviate()
collection = weaviate_client.collections.get("medibot")
embeddings = initialize_embeddings()

# Token ve istek sayısı limitleri
MAX_TOKENS_PER_REQUEST = 500  # Her istekte maksimum token
DAILY_REQUEST_LIMIT = 50      # Günlük maksimum istek sayısı
HOURLY_REQUEST_LIMIT = 10     # Saatlik maksimum istek sayısı

class UsageTracker:
    def __init__(self):
        self.daily_requests = 0
        self.hourly_requests = 0
        self.last_reset = datetime.now()
        self.hour_reset = datetime.now()
    
    def can_make_request(self):
        now = datetime.now()
        
        # Günlük sayacı sıfırla
        if (now - self.last_reset) > timedelta(days=1):
            self.daily_requests = 0
            self.last_reset = now
        
        # Saatlik sayacı sıfırla
        if (now - self.hour_reset) > timedelta(hours=1):
            self.hourly_requests = 0
            self.hour_reset = now
        
        if self.daily_requests >= DAILY_REQUEST_LIMIT:
            return False, "Günlük soru limitine ulaşıldı. Yarın tekrar deneyin."
        
        if self.hourly_requests >= HOURLY_REQUEST_LIMIT:
            minutes_to_wait = 60 - (now - self.hour_reset).minutes
            return False, f"Saatlik limit aşıldı. {minutes_to_wait} dakika sonra tekrar deneyin."
        
        return True, ""
    
    def increment_counters(self):
        self.daily_requests += 1
        self.hourly_requests += 1

def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def check_medical_context_safety(query: str, context: str) -> tuple[bool, str, list[str]]:
    """
    Kontekst ve soru arasındaki uyumu ve güvenliği kontrol eder.
    Returns:
        - is_safe: Yanıtın güvenli olup olmadığı
        - warning: Varsa uyarı mesajı
        - alternative_topics: Alternatif konu önerileri
    """
    # Hassas tıbbi konuları tanımlayan anahtar kelimeler
    sensitive_topics = {
        "abortion": ["pregnancy", "family planning", "reproductive health"],
        "suicide": ["mental health support", "depression treatment", "professional help"],
        "terminal illness": ["treatment options", "pain management", "supportive care"],
        "addiction": ["recovery programs", "substance abuse treatment", "professional support"],
    }
    
    query_lower = query.lower()
    context_lower = context.lower()
    
    # Kontekst ile soru arasında uyumsuzluk kontrolü
    context_keywords = set(context_lower.split())
    query_keywords = set(query_lower.split())
    context_relevance = len(context_keywords.intersection(query_keywords)) / len(query_keywords)
    
    if context_relevance < 0.2:  # Eğer sorununun %20'sinden azı kontekstte varsa
        return False, "Bu soru için yeterli tıbbi bilgi kontekstte bulunmamaktadır.", []

    # Hassas konuların kontrolü
    for topic, alternatives in sensitive_topics.items():
        if topic in query_lower or topic in context_lower:
            return False, f"Bu konu hassas tıbbi bilgiler içermektedir. Alternatif olarak şu konuları araştırabilirsiniz:", alternatives
    
    return True, "", []

def get_ai_response(query: str, context: str, use_gemini: bool = True):
    # Önce güvenlik kontrolü yap
    is_safe, warning, alternatives = check_medical_context_safety(query, context)
    
    if not is_safe:
        response = warning + "\n"
        if alternatives:
            response += "\nÖnerilen alternatif konular:\n"
            response += "\n".join(f"- {alt}" for alt in alternatives)
        return response

    if use_gemini:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""You are a medical information assistant. Please follow these guidelines:
1. Only provide information that is directly supported by the given medical context
2. If the question goes beyond the provided context, politely decline to answer
3. For sensitive topics, suggest discussing with a healthcare provider
4. Be clear about the limitations of the information

Medical Context:
{context}

Question:
{query}

Please provide a response following the above guidelines."""
        
        response = model.generate_content(prompt)
        return response.text
    else:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a medical information assistant. Please follow these guidelines:
1. Only provide information that is directly supported by the given medical context
2. If the question goes beyond the provided context, politely decline to answer
3. For sensitive topics, suggest discussing with a healthcare provider
4. Be clear about the limitations of the information"""},
                {"role": "user", "content": f"Medical Context:\n{context}\n\nQuestion:\n{query}"},
            ],
            max_tokens=MAX_TOKENS_PER_REQUEST,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

def ask_medical_question():
    client = connect_to_weaviate()
    collection = client.collections.get("medibot")
    embeddings = initialize_embeddings()
    usage_tracker = UsageTracker()
    use_gemini = True  # Varsayılan olarak OpenAI kullanılacak

    while True:
        print("\n=== Ask a Medical Question ===")
        print("(Type 'quit' or 'exit' to end the program)")
        print("(Type 'switch' to toggle between OpenAI and Gemini)")
        print(f"Current AI: {'Gemini' if use_gemini else 'OpenAI'}")
        print(f"Günlük kalan soru hakkı: {DAILY_REQUEST_LIMIT - usage_tracker.daily_requests}")
        print(f"Saatlik kalan soru hakkı: {HOURLY_REQUEST_LIMIT - usage_tracker.hourly_requests}")
        
        query = input("Your question: ").strip().lower()
        
        if query in ['quit', 'exit']:
            print("\nTeşekkürler! Program sonlandırılıyor...")
            break
            
        if query == 'switch':
            use_gemini = not use_gemini
            print(f"\nSwitched to {'Gemini' if use_gemini else 'OpenAI'}")
            continue
            
        if not query:
            print("Lütfen bir soru girin.")
            continue

        # Kullanım limitlerini kontrol et
        can_request, limit_message = usage_tracker.can_make_request()
        if not can_request:
            print(f"\n⚠️ {limit_message}")
            continue

        vector = embeddings.embed_query(query)
        results = collection.query.near_vector(near_vector=vector, limit=1)

        if results and results.objects:
            obj = results.objects[0]
            source = obj.properties.get('source')
            page = obj.properties.get('page')
            context = obj.properties.get('content')

            # Token sayısını kontrol et
            total_tokens = count_tokens(context + query)
            if total_tokens > MAX_TOKENS_PER_REQUEST:
                context = context[:int(len(context) * (MAX_TOKENS_PER_REQUEST / total_tokens))]

            print(f"\n📄 Source: {source} (Page {page})")
            print(f"📚 Content: {context[:500]}...")
            
            try:
                answer = get_ai_response(query, context, use_gemini)
                print(f"\n💬 Answer:\n{answer}")
                
                # Başarılı istek sonrası sayaçları artır
                usage_tracker.increment_counters()
                
            except Exception as e:
                print(f"\n⚠️ Bir hata oluştu: {str(e)}")
                continue
        else:
            print("Bu soru için uygun bir kaynak bulunamadı.")

    client.close()
    weaviate_client.close()

if __name__ == "__main__":
    ask_medical_question()
