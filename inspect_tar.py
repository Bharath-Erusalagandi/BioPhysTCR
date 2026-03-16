import tarfile

tar_path = 'ImmuneCODE-Repertoires-002.2.tgz'

try:
    with tarfile.open(tar_path, "r:gz") as tar:
        print("First 10 members in tar:")
        for member in tar.getmembers()[:10]:
            print(member.name)
            if member.isfile():
                f = tar.extractfile(member)
                if f:
                    print(f"--- Content of {member.name} (first 200 bytes) ---")
                    print(f.read(200))
                    print("-----------------------------------------------")
except Exception as e:
    print(f"Error reading tar: {e}")
