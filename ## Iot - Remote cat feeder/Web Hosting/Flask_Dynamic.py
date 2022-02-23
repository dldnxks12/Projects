from flask import Flask

app =Flask(__name__)

@app.route("/led/<state>") # 산형 괄호 <> 안에 인자를 반영해서 호출
def led(state): # 인자 반드시 받아야함
  return state

if __name__ == "__main__":
  return app.run()
