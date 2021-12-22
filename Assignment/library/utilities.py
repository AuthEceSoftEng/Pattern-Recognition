import requests
import pandas as pd

# Get info about a specific user/organization by his/her/its username
def getUser(username):
    url = "https://api.github.com/users/" + username
    r = requests.get(url = url, params = {}) 
    response = r.json() 
    print(response["repos_url"])

# Get the public repositories of a user/organization
def getUserRepos(username):
    url = "https://api.github.com/users/" + username + "/repos"
    r = requests.get(url = url, params = {}) 
    response = r.json() 
    print(response[0].keys())
    # for repo in response:
    #     print(repo["name"])
    #     print(repo["language"])

# Get the information of a public repository
def getRepoInfo(repoOwner, repoName):
    url = "https://api.github.com/repos/" + repoOwner + "/" + repoName
    r = requests.get(url = url, params = {}) 
    response = r.json() 
    print(response)

# Get basic information about the most popular repositories
def getMostPopularRepositoriesInfos():
    repos = pd.read_csv("./mostPopularRepositories.csv", sep=";")
    for _, row in repos.iterrows():
        url = "https://api.github.com/repos/" + row.RepositoryOwner + "/" + row.RepositoryName
        r = requests.get(url = url, params = {}) 
        response = r.json() 
        if "topics" in response:
            print(response["topics"])

# Get the comments in issues of a public repository
def getRepoIssuesComments(repoOwner, repoName):
    url = "https://api.github.com/repos/" + repoOwner + "/" + repoName + "/issues/comments"
    r = requests.get(url = url, params = {}) 
    response = r.json() 
    print(response)

# Get the releases of a public repository
def getRepoReleases(repoOwner, repoName):
    url = "https://api.github.com/repos/" + repoOwner + "/" + repoName + "/releases"
    r = requests.get(url = url, params = {}) 
    response = r.json() 
    print(response)

# Get a specific commit of a public repository
def getRepoCommit(repoOwner, repoName, commitSHA):
    url = "https://api.github.com/repos/" + repoOwner + "/" + repoName + "/commits/" + commitSHA
    r = requests.get(url = url, params = {}) 
    response = r.json() 
    print(response)

# getUser("karanikiotis")
# getUserRepos("AuthEceSofteng")
# getRepoInfo("AuthEceSofteng", "emb-ntua-workshop")
# getMostPopularRepositoriesInfos()
# getRepoIssuesComments("AuthEceSofteng", "emb-ntua-workshop")
# getRepoReleases("AuthEceSofteng", "emb-ntua-workshop")
# getRepoCommit("AuthEceSofteng", "emb-ntua-workshop", "41e03e26db38caf3d2b9c500d56be1a1327d8c84")
