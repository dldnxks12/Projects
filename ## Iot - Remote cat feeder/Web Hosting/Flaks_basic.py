from flask imort Flask

app = Flask(__name__)


# function Deco를 통해 알맞는 주소로 안내 
@app.route("/")  # host 주소 + "/"로 해당 Web Loading
def helloworld():
  return "hello world"

if __name__ == "__main__":
  app.run(host = "0.0.0.0/8080")
