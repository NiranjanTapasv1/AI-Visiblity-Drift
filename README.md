# GEO Drift Tracker

GEO Drift Tracker is a simple way to see how AI answers move around when the same question is asked more than once.

It helps you spot when a brand looks strong in one answer but weak in another. It also shows when two AI models disagree with each other. That matters because one good looking answer is not always a reliable signal.

## What this project does

The app asks the same question several times, sends it to Gemini and Groq, and then compares the answers.

It looks at:

1. Which brands appear most often
2. Where those brands usually appear in the answer
3. How much the answers change from run to run
4. Where the two models disagree most

The result is turned into a dashboard that is easier to read than raw model output.

## Why it matters

AI answers can sound confident even when they are inconsistent. A brand may look strong in one run and disappear in the next. This project helps you see that pattern clearly.

For the PEEC team, this means you can use the app to understand:

1. Which brands are more stable
2. Which brands are worth watching more closely
3. Where Gemini and Groq are not telling the same story

## What you will see

The app has a home page and a dashboard.

The home page explains the idea in plain language.

The dashboard shows:

1. A summary of the latest run
2. A stability ranking table
3. Charts that show the main patterns
4. A short takeaway section that explains the result in simple words
5. A raw data section for anyone who wants the full output

## How it works

The flow is simple.

1. Choose a prompt such as best CRM tools for startups
2. Run that prompt through Gemini and Groq several times
3. Optionally ground the run with live web search context
4. Compare the answers
5. Score the stability of each brand
6. Show the final result in the dashboard

## What the numbers mean

1. Mentioned means how often a brand showed up
2. Avg position means where it usually appeared in the answer
3. Stability means how steady the brand was across repeated runs
4. Drift gap means how far apart Gemini and Groq were

Lower average position is better because it means the brand appears closer to the top of the answer.

## Tech stack

1. Python
2. Streamlit
3. Pandas
4. Matplotlib
5. Gemini API
6. Groq API

## Local setup

1. Activate your environment

```bash
conda activate geo_drift
```

2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

3. Create your environment file

```bash
cp .env.example .env
```

4. Add your API keys to `.env`

```bash
GEMINI_API_KEY=
GROQ_API_KEY=
```

If the keys are missing, the app still runs with fallback data so you can test the interface.

## Run locally

Start the Streamlit app with:

```bash
/opt/anaconda3/envs/geo_drift/bin/streamlit run app.py
```

If you want to run the backend analysis directly, use:

```bash
python src/geo_drift_tracker.py
```

## Deploying on GitHub and Streamlit Community Cloud

1. Create a new GitHub repository and push this project to it.
2. Make sure `README.md` is in the root of the repository.
3. Keep `.env` out of GitHub.
4. Add the same values from `.env` to Streamlit Community Cloud Secrets.
5. Deploy the app from the GitHub repository in Streamlit Community Cloud.
6. Copy the live Streamlit URL and paste it here:

```text
Live app: (AI Visiblity Tracker)(https://ai-visiblity-drift-8t9ywwcijebgn3pohxfhhx.streamlit.app/)
```

Streamlit Community Cloud connects directly to GitHub repositories and lets you deploy an app in a few minutes. It also uses a secrets system so you can keep API keys out of the repository. [Streamlit Community Cloud docs](https://docs.streamlit.io/deploy/streamlit-community-cloud) [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) [Secrets management](https://docs.streamlit.io/develop/concepts/connections/secrets-management)

GitHub repositories can be created from the web interface in a few clicks. [GitHub new repository docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)

## Output files

The app writes analysis files into the `results` folder.

You may see:

1. CSV files for each provider
2. A combined comparison table
3. A short summary file
4. A markdown report with the main findings

## Short version

GEO Drift Tracker shows whether an AI answer is stable or just lucky.

It helps you tell the difference between a real signal and a one time result.
