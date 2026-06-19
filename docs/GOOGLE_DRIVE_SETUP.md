# Google Drive Setup for Site Survey App

## Overview

This guide walks through setting up Google Drive integration for the site survey app. The app uses a Google service account to authenticate with Drive, allowing it to upload and process images in a shared team folder.

## One-Time Setup (Admin Only)

### 1. Create Google Cloud Project

1. Go to https://console.cloud.google.com/
2. Click "Select a Project" at the top
3. Click "NEW PROJECT"
4. Enter name: "Site Survey"
5. Click "CREATE"
6. Wait for the project to be created

### 2. Enable Google Drive API

1. In the Google Cloud Console, search for "Google Drive API"
2. Click the search result
3. Click "ENABLE"
4. You should see "Google Drive API is now enabled"

### 3. Create Service Account

1. In the left menu, go to "Service Accounts"
2. Click "CREATE SERVICE ACCOUNT"
3. Service account name: `site-survey-app`
4. Click "CREATE AND CONTINUE"
5. Grant role: "Editor"
6. Click "CONTINUE"
7. Click "DONE"

### 4. Create and Download JSON Key

1. In Service Accounts list, click on `site-survey-app`
2. Go to "KEYS" tab
3. Click "ADD KEY" → "Create new key"
4. Choose "JSON"
5. Click "CREATE"
6. A JSON file will download automatically
7. Keep this file safe — it's your credentials

### 5. Share Google Drive Folder with Service Account

1. Open the team folder: https://drive.google.com/drive/folders/1FXXNVLaAFWSc1HYDUx8lyaosqF9BJdgL
2. Right-click the folder → "Share"
3. Get the service account email from the JSON key file (looks like `site-survey-app@...iam.gserviceaccount.com`)
4. Paste the email in the share dialog
5. Grant "Editor" access
6. Click "Share"

## Configure Streamlit Secrets

### Local Development

1. Copy `.streamlit/secrets.toml.template` to `.streamlit/secrets.toml`
2. Open the JSON key file you downloaded
3. Copy the entire JSON content (between the curly braces)
4. In `secrets.toml`, replace the `GOOGLE_DRIVE_CREDENTIALS` value with your actual JSON (keep it as a string)
5. Save and restart your Streamlit app

### Streamlit Cloud Deployment

1. Go to https://share.streamlit.io/
2. Select your app → "Settings"
3. Scroll to "Secrets"
4. Paste the entire contents of your local `.streamlit/secrets.toml` file
5. Click "Save"
6. Redeploy the app

## Verify Setup

After configuration:

1. Run the app locally or deploy to Streamlit Cloud
2. Upload a test image
3. Check that a new folder appears in your Google Drive
4. Verify the folder structure was created (01_Raw_Images, 02_Processed_Sites, etc.)

## Troubleshooting

- **"Google Drive credentials not configured"** — Verify `GOOGLE_DRIVE_CREDENTIALS` is in secrets.toml with the full JSON content
- **"Failed to authenticate"** — Check the JSON key is valid and hasn't been revoked
- **"Permission denied"** — Ensure the service account was shared with the team folder with Editor access
- **Images not uploading** — Check app logs for API errors

## Team Access

Once configured, both team members can:
- Access the app at https://site-survey.streamlit.app/
- Upload images, which go to Google Drive
- See all processed files and reports
- No additional configuration needed — they just use the app

All files are stored in the shared Google Drive folder accessible to the entire team.
