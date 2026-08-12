import requests
import re
from dotenv import load_dotenv
load_dotenv()
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def fetch_pr_diff(pr_url:str)->dict:
    """Given a public PR Url, fetching the raw diff and parsing it into
    individual file changes"""

    #convert raw url to raw diff URL, means place .diff suffix if not present
    if not pr_url.endswith(".diff"):
        diff_url = f"{pr_url}.diff"
    else:
        diff_url = pr_url

    #fetch raw diff from github
    headers = {"Accept":"application/vnd.github.v3.diff"}
    #above tells GitHub’s API to return the raw code diff of a Pull Request or Commit 
    # instead of the standard JSON metadata

    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        response = requests.get(diff_url, headers=headers)

        #if response succeeds it return success code i.e 200, if it does'nt we throw an exception
        if response.status_code !=200:
            print(f"Error in fetching PR diff! Error Code: {response.status_code}\n\n")
            raise RuntimeError("Error in fetching PR diff!")


        #raw diff: (texts of pull request containing changes with file paths on top)
        raw_diff = response.text
        
        #splitting by diff --git , because it marks the beginning of each file, 
        # "diff --git will be cut out in the file_diffs, so we get the paths of files"
        file_diffs = raw_diff.split("diff --git ")

        parsed_files = []

        #below we are extracting the file paths from file_diffs (list)
        #but we also check if list after splitting contains any empty "" data, because when we split by diff --git
        #so the first element in the list after split would be an empty "", therefore we continue to skip it
        for file_block in file_diffs:
            if not file_block.strip():
                continue

            #extract file path using regex
            reg_match = re.search(r"^a/(\S+)\s+b/(\S+)", file_block) #fetches the original path, and new path, we need the new one
            file_path = reg_match.group(2) if reg_match else "unknown_file"  #group(2) has the b/... updated file
            
            parsed_files.append({
                "file_path": file_path,
                "raw_diff_content": "diff --git" + file_block
            })

        return {
            "raw_diff": raw_diff,
            "files_changed": parsed_files,
        }

    except Exception as e:
        print(f"\n\nException occurred in pull_req_files.py:\n{e}")
        raise RuntimeError(f"Exception occurred while fetching PR Diff\n{e}")