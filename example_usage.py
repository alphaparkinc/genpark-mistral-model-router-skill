from client import MistralModelRouterClient
client = MistralModelRouterClient()

# Route coding task
r1 = client.route_and_infer("coding", "Write a Python async REST API with FastAPI and PostgreSQL", "sovereign")
print(f"Model: {r1['selected_model']} | Cost: ${r1['cost_per_1m_tokens']}/1M tokens")
print(f"Response: {r1['response'][:100]}...")

# Route voice transcription
r2 = client.route_and_infer("voice", "Transcribe this meeting audio in real-time", "cloud")
print(f"Model: {r2['selected_model']} | Cost: ${r2['cost_per_1m_tokens']}/1M tokens")
