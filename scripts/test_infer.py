import httpx, json, sys

def health_check():
    try:
        resp = httpx.get('http://127.0.0.1:9000/health')
        print('Health check status:', resp.status_code)
        print('Body:', resp.json())
    except Exception as e:
        print('Health check error:', e)
        sys.exit(1)

def infer():
    payload = {
        "prompt": "Hello world",
        "profile": "interactive",
        "max_output_tokens": 5,
        "policy": "fifo"
    }
    try:
        with httpx.Client(timeout=None) as client:
            with client.stream('POST', 'http://127.0.0.1:9000/infer', json=payload) as resp:
                print('Infer response status:', resp.status_code)
                if resp.status_code != 200:
                    print('Error body:', resp.read().decode('utf-8'))
                    return
                print('Streaming tokens:')
                for line in resp.iter_lines():
                    if line:
                        print(line)
    except Exception as e:
        print('Infer error:', e)
        sys.exit(1)

if __name__ == '__main__':
    health_check()
    infer()
