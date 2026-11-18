"""Local OpenAI-compatible text embedding endpoint for smoke evaluation."""
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(4)
model = SentenceTransformer(str(Path.home()/'.cache/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master'), device='cpu', local_files_only=True)

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            texts=body['input']
            if isinstance(texts,str): texts=[texts]
            if body.get('model') != 'Qwen3-Embedding-0.6B': raise ValueError('Unknown model')
            if not all(isinstance(t,str) for t in texts): raise ValueError('Text input required')
            if any(len(model.tokenizer.encode(t)) > model.max_seq_length for t in texts): raise ValueError('Input exceeds model context')
            vectors=model.encode(texts,batch_size=2,normalize_embeddings=True).tolist()
            usage_path=Path(__file__).resolve().parents[1]/'outputs/mobilemem-qwen/smoke20/embedding_usage.jsonl'
            usage_path.parent.mkdir(parents=True,exist_ok=True)
            with usage_path.open('a',encoding='utf-8') as log:
                log.write(json.dumps({'timestamp':time.time(),'model':body['model'],'texts':len(texts),'input_tokens':sum(len(model.tokenizer.encode(t)) for t in texts),'dimensions':len(vectors[0]),'api_cost':0})+'\n')
            result={'object':'list','model':body['model'],'data':[{'object':'embedding','index':i,'embedding':v} for i,v in enumerate(vectors)],'usage':{'prompt_tokens':sum(len(model.tokenizer.encode(t)) for t in texts),'total_tokens':sum(len(model.tokenizer.encode(t)) for t in texts)}}
            payload=json.dumps(result).encode()
            self.send_response(200)
        except Exception as exc:
            payload=json.dumps({'error':str(exc)}).encode()
            self.send_response(400)
        self.send_header('Content-Type','application/json')
        self.end_headers()
        self.wfile.write(payload)

print('Qwen embedding ready on 127.0.0.1:8008',flush=True)
HTTPServer(('127.0.0.1',8008),Handler).serve_forever()
