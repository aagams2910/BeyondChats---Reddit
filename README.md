# BeyondChats---Reddit

## Reddit User Persona Generator

This tool generates a detailed persona for any Reddit user by analyzing their latest posts and comments using the Gemini API.

### Features
- Scrapes latest posts and comments from a Reddit user
- Uses Gemini 1.5 Flash API to generate a detailed persona
- Outputs a markdown-formatted `.txt` file with citations

### Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with the following keys:
   ```env
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=your_user_agent
   GEMINI_API_KEY=your_gemini_api_key
   ```

### Usage
```bash
python main.py https://www.reddit.com/user/<username>/
```

The output will be saved as `<username>_persona.txt` in the current directory.