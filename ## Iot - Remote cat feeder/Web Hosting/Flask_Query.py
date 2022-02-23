from flask import Flask, request # Query 동작을 위해 request import 

app = Flask(__name__)

# Web Brower에서 localhost:5000/led?state=off 와 같이 Query 메세지와 함께 입력 

@app.route("/led")
def led():
  state = request.args.get("state")
  if state == "on":
    return "LED On"
  else:
    return "LED Off"
  

if __name__ == "__main__":
  app.run(host = "0.0.0.0/8080") # Netis Private Ip 
    
