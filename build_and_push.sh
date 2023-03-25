sudo aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 040045840992.dkr.ecr.ap-southeast-1.amazonaws.com
sudo docker build -t multitracker .
sudo docker tag multitracker:latest 040045840992.dkr.ecr.ap-southeast-1.amazonaws.com/multitracker:latest
sudo docker push 040045840992.dkr.ecr.ap-southeast-1.amazonaws.com/multitracker:latest