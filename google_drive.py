import os
import json
import io
from google.oauth2.service_account import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
import streamlit as st

class GoogleDriveManager:
    """Manages Google Drive operations: auth, file upload/download, folder creation."""

    def __init__(self, credentials_json_str):
        """
        Initialize with credentials JSON string from Streamlit secrets.

        Args:
            credentials_json_str: JSON string of service account credentials
        """
        try:
            creds_dict = json.loads(credentials_json_str)
            self.credentials = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            self.service = build('drive', 'v3', credentials=self.credentials)
        except Exception as e:
            raise ValueError(f"Failed to initialize Google Drive: {e}")

    def get_or_create_folder(self, parent_folder_id, folder_name):
        """
        Get folder by name in parent, or create if doesn't exist.

        Args:
            parent_folder_id: Google Drive folder ID of parent
            folder_name: Name of folder to find/create

        Returns:
            Folder ID (string)
        """
        escaped_name = folder_name.replace("'", "\\'")
        query = f"'{parent_folder_id}' in parents and name='{escaped_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"

        results = self.service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)',
            pageSize=1
        ).execute()

        if 'error' in results:
            raise ValueError(f"Failed to list folders: {results['error']}")

        files = results.get('files', [])
        if files:
            return files[0]['id']

        # Create folder
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }
        folder = self.service.files().create(
            body=file_metadata,
            fields='id'
        ).execute()

        if 'error' in folder:
            raise ValueError(f"Failed to create folder: {folder['error']}")

        return folder.get('id')

    def upload_file(self, file_path, folder_id, file_name=None):
        """
        Upload file to Google Drive folder.

        Args:
            file_path: Local file path
            folder_id: Google Drive folder ID
            file_name: Optional custom name for file in Drive

        Returns:
            File ID (string)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = file_name or os.path.basename(file_path)

        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }

        media = MediaFileUpload(file_path, resumable=True)
        file_obj = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        if 'error' in file_obj:
            raise ValueError(f"Failed to upload file: {file_obj['error']}")

        return file_obj.get('id')

    def upload_bytes(self, file_bytes, folder_id, file_name, mime_type='application/octet-stream'):
        """
        Upload bytes to Google Drive.

        Args:
            file_bytes: Bytes to upload
            folder_id: Google Drive folder ID
            file_name: Name of file in Drive
            mime_type: MIME type

        Returns:
            File ID (string)
        """
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }

        media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mime_type, resumable=True)
        file_obj = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        if 'error' in file_obj:
            raise ValueError(f"Failed to upload file: {file_obj['error']}")

        return file_obj.get('id')

    def download_file(self, file_id, local_path):
        """
        Download file from Google Drive to local path.

        Args:
            file_id: Google Drive file ID
            local_path: Local destination path
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            with open(local_path, 'wb') as f:
                while True:
                    status, done = request.next_chunk()
                    if done:
                        break
        except Exception as e:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise IOError(f"Failed to download file: {e}")

        return local_path

    def list_files(self, folder_id, file_type='all'):
        """
        List files in folder.

        Args:
            folder_id: Google Drive folder ID
            file_type: 'folders', 'images', or 'all'

        Returns:
            List of file dicts with id, name, mimeType
        """
        if file_type == 'folders':
            query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        elif file_type == 'images':
            query = f"'{folder_id}' in parents and (mimeType contains 'image/') and trashed=false"
        else:
            query = f"'{folder_id}' in parents and trashed=false"

        results = self.service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, mimeType, size)',
            pageSize=100
        ).execute()

        if 'error' in results:
            raise ValueError(f"Failed to list files: {results['error']}")

        return results.get('files', [])

def get_drive_manager():
    """Get authenticated Drive manager from Streamlit secrets."""
    credentials_json = st.secrets.get('GOOGLE_DRIVE_CREDENTIALS')
    if not credentials_json:
        raise ValueError("Google Drive credentials not configured in Streamlit secrets.")
    return GoogleDriveManager(credentials_json)
