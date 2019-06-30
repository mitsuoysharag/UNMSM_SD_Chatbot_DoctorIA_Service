# Chatbot_service

## DEPLOY TO HEROKU

### NECESARY FILES
* Create Procfile file
* Create nltk.txt, if you use nltk library
* Create a file with the libraries that you are using in the program, if you ya want to use all the libraries that you have installed used the next command in the cmd:
`pip freeze > requirements.txt`

### COMMANDS
* `heroku login`
* `heroku create`  --If you haven´t created a heroku repository yet.
* `git init`
* `heroku git:remote -a <<heroku_repo_name>>`
* `git commit -am “initial commit”`
* `git push heroku master `
