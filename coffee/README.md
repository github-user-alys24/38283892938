Built with K8s cluster
> python files utilize flask to allow for /POST and /GET functionality and integration with services to host locally on local machine
> services to enable communication between different parts of the application 
>> utilizing ClusterIP to host locally
> ports 8080:80 were mainly used

User's guide
BEFORE RUNNING, change the file directories in the run.sh file
# Running with run.sh
##### In WSL command line,
cd /mnt/c/Users/your-user/your-file-path/coffee
./run.sh
##### If unable to run ./run.sh directly, download dos2unix
sudo apt-get install dos2unix
./run.sh 					# Runs the entire deployment
##### Access the application through localhost:8081

Localhost routes
website > localhost:8081
preprocessing > localhost:8080
inference > localhost:8083
prediction > localhost:8084


