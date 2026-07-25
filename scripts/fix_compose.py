import re

with open("docker-compose.yml") as f:
    content = f.read()

# Remove conflict markers, keep both blocks
pattern = r'<<<<<<< Updated upstream\n(.*?)=======\n(.*?)>>>>>>> Stashed changes'
def replace(m):
    return m.group(1).strip() + "\n" + m.group(2).strip()

content = re.sub(pattern, replace, content, flags=re.DOTALL)

with open("docker-compose.yml", "w") as f:
    f.write(content)

print("docker-compose.yml resolved - kept both pid:host and extra_hosts")