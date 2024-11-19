import json

def scrape_linkedin_profile(url: str, mock: bool = True) -> dict[str, str]:
    print("provided url:", url)

    if not mock:
        raise NotImplementedError
    
    with open("data/linkedin.json") as f:
        raw_data = json.load(f)

    blacklist = ["people_also_viewed", "certifications"]

    data = {
        k: v
        for k, v in raw_data.items()
        if v not in ([], "", None) and k not in blacklist
    }

    if groups := data.get("groups"):
        for group in groups:
            group.pop("profile_pic_url")

    return data


if __name__ == '__main__':
    print(scrape_linkedin_profile("https://www.linkedin.com/in/eden-marco"))
