from huggingface_hub import login, upload_file
from huggingface_sb3 import push_to_hub

login()

repo_id = "asalgad2/rl_lunar_lander"

push_to_hub(
    repo_id=repo_id,
    filename="ppo-LunarLander-v2_2.zip",
    commit_message="commit lunar lander rl model",
)

upload_file(
    repo_id=repo_id,
    path_or_fileobj="replay.mp4",
    path_in_repo="replay.mp4",
    commit_message="add replay video",
)

upload_file(
    repo_id=repo_id,
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    commit_message="update README",
)
