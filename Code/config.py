# Paths and URLs
PATH = "C:\\Users\\arifr\\OneDrive\\Desktop\\chromedriver.exe"
url1 = "https://chromedino.com"
url2 = "chrome://dino/"
loss_file_path = "./model/loss_df.csv"

# Java scripts
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'; Cloud.config.WIDTH = 0"
init_script2 = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas';"
getScreenScript = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL(" \
                  ").substring(22)"
