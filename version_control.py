#!/usr/bin/env python3
"""
S3-Based Version Control System for EMR Workspaces
A lightweight version control system using S3 with encryption
"""

import boto3
import os
import json
import hashlib
import datetime
import zipfile
import io
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class S3VersionControl:
    def __init__(self, bucket_name, passphrase, current_workspace_id):
        """
        Initialize S3 Version Control
        
        Args:
            bucket_name: S3 bucket name for storing code
            passphrase: Your secret passphrase for encryption
            current_workspace_id: Unique identifier for the *current* workspace.
                                  This will be recorded as metadata for commits.
        """
        self.bucket_name = bucket_name
        self.current_workspace_id = current_workspace_id # Renamed for clarity
        self.s3_client = boto3.client('s3')
        
        # Generate encryption key from passphrase
        self.cipher_suite = self._generate_cipher(passphrase)
        
        # Base paths in S3 for the SINGLE, SHARED repository
        # This is the key change: no workspace_id here
        self.base_repo_path = "emr-shared-code-repo" # Central repository path
        self.versions_path = f"{self.base_repo_path}/versions"
        self.metadata_path = f"{self.base_repo_path}/metadata"
        self.branches_path = f"{self.base_repo_path}/branches" # Pointer for branches

    def _generate_cipher(self, passphrase):
        """Generate Fernet cipher from passphrase"""
        password = passphrase.encode()
        salt = b'salt_for_consistency'  # In production, use random salt stored securely
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    def _encrypt_data(self, data):
        """Encrypt data using the cipher"""
        if isinstance(data, str):
            data = data.encode()
        return self.cipher_suite.encrypt(data)
    
    def _decrypt_data(self, encrypted_data):
        """Decrypt data using the cipher"""
        return self.cipher_suite.decrypt(encrypted_data)
    
    def _calculate_hash(self, data):
        """Calculate SHA256 hash of data"""
        return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()
    
    def _create_archive(self, source_path):
        """Create zip archive of source directory"""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            source_path = Path(source_path)
            
            if source_path.is_file():
                zip_file.write(source_path, source_path.name)
            else:
                for file_path in source_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(source_path)
                        zip_file.write(file_path, relative_path)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def commit(self, source_path, commit_message, branch="main"):
        """
        Commit code to the shared S3 repository
        
        Args:
            source_path: Path to code directory or file
            commit_message: Commit message
            branch: Branch name (default: main)
        """
        try:
            # Create archive
            archive_data = self._create_archive(source_path)
            
            # Generate version info
            timestamp = datetime.datetime.now().isoformat()
            version_hash = self._calculate_hash(archive_data)
            
            # Create metadata (now includes 'committer_workspace_id')
            metadata = {
                'committer_workspace_id': self.current_workspace_id, # This workspace made the commit
                'timestamp': timestamp,
                'commit_message': commit_message,
                'branch': branch,
                'hash': version_hash,
                'size': len(archive_data)
            }
            
            # Encrypt archive
            encrypted_archive = self._encrypt_data(archive_data)
            
            # Upload encrypted archive to the shared versions path
            version_key = f"{self.versions_path}/{branch}/{version_hash}.zip.enc"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=version_key,
                Body=encrypted_archive
            )
            
            # Upload metadata to the shared metadata path
            metadata_key = f"{self.metadata_path}/{branch}/{version_hash}.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2)
            )
            
            # Update branch pointer in the shared branches path
            branch_key = f"{self.branches_path}/{branch}.json"
            branch_info = {
                'latest_hash': version_hash,
                'latest_timestamp': timestamp,
                'last_committed_by_workspace_id': self.current_workspace_id # Record who last updated the branch pointer
            }
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=branch_key,
                Body=json.dumps(branch_info, indent=2)
            )
            
            print(f"✅ Committed successfully to shared repository!")
            print(f"    Version: {version_hash[:8]}")
            print(f"    Branch: {branch}")
            print(f"    Message: {commit_message}")
            print(f"    Timestamp: {timestamp}")
            print(f"    Committed by Workspace: {self.current_workspace_id}")
            
            return version_hash
            
        except Exception as e:
            print(f"❌ Error during commit: {e}")
            return None
    
    def checkout(self, target_path, branch="main", version_hash=None):
        """
        Checkout code from the shared S3 repository
        
        Args:
            target_path: Path where to extract code
            branch: Branch to checkout (default: main)
            version_hash: Specific version hash (optional, defaults to latest)
        """
        try:
            # Get version hash if not provided (from the shared branch pointer)
            if not version_hash:
                version_hash = self._get_latest_version(branch)
                if not version_hash:
                    print(f"❌ No versions found for branch '{branch}' in the shared repository")
                    return False
            
            # Download encrypted archive from the shared versions path
            version_key = f"{self.versions_path}/{branch}/{version_hash}.zip.enc"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=version_key)
            encrypted_data = response['Body'].read()
            
            # Decrypt archive
            decrypted_data = self._decrypt_data(encrypted_data)
            
            # Extract archive
            target_path = Path(target_path)
            target_path.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(io.BytesIO(decrypted_data), 'r') as zip_file:
                zip_file.extractall(target_path)
            
            print(f"✅ Checkout successful from shared repository!")
            print(f"    Version: {version_hash[:8]}")
            print(f"    Branch: {branch}")
            print(f"    Target: {target_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during checkout: {e}")
            return False
    
    def _get_latest_version(self, branch):
        """Get latest version hash for a branch from the shared branch pointer"""
        try:
            branch_key = f"{self.branches_path}/{branch}.json"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=branch_key)
            branch_info = json.loads(response['Body'].read())
            return branch_info['latest_hash']
        except Exception as e:
            print(f"Warning: Could not retrieve latest version for branch '{branch}'. {e}")
            return None
    
    def list_versions(self, branch="main", filter_by_workspace_id=None):
        """
        List all versions in a branch from the shared repository.
        Can optionally filter by the workspace ID that made the commit.
        """
        try:
            prefix = f"{self.metadata_path}/{branch}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                print(f"No versions found for branch '{branch}' in the shared repository.")
                return []
            
            versions = []
            for obj in response['Contents']:
                # Get metadata
                metadata_response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=obj['Key']
                )
                metadata = json.loads(metadata_response['Body'].read())
                
                # Apply workspace filter if provided
                if filter_by_workspace_id and metadata.get('committer_workspace_id') != filter_by_workspace_id:
                    continue
                
                versions.append(metadata)
            
            # Sort by timestamp (newest first)
            versions.sort(key=lambda x: x['timestamp'], reverse=True)
            
            print(f"📋 Version History for branch '{branch}' (Shared Repository):")
            if filter_by_workspace_id:
                print(f"(Filtered by Workspace ID: {filter_by_workspace_id})")
            print("-" * 80)
            for version in versions:
                print(f"Hash: {version['hash'][:8]}")
                print(f"Date: {version['timestamp']}")
                print(f"Message: {version['commit_message']}")
                print(f"Committed by Workspace: {version.get('committer_workspace_id', 'N/A')}") # Use .get for robustness
                print("-" * 40)
            
            return versions
            
        except Exception as e:
            print(f"❌ Error listing versions: {e}")
            return []
    
    # The 'sync_from_workspace' method is no longer needed in its original form
    # as all workspaces will interact with the single shared repository.
    # If you need to "sync" in the new model, it's just a 'checkout' of the latest.
    def pull_latest(self, target_path, branch="main"):
        """
        Pulls the latest version from the shared repository for a given branch.
        This effectively replaces the 'sync_from_workspace' logic.
        """
        print(f"Attempting to pull latest for branch '{branch}' to '{target_path}'...")
        return self.checkout(target_path, branch)

# Usage examples and CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='S3 Version Control System for Shared EMR Workspaces')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    # Renamed from --workspace to --current-workspace-id for clarity in shared repo model
    parser.add_argument('--current-workspace-id', required=True, help='ID of the current EMR Workspace interacting with the shared repo')
    parser.add_argument('--passphrase', required=True, help='Encryption passphrase')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Commit command
    commit_parser = subparsers.add_parser('commit', help='Commit code to the shared repository')
    commit_parser.add_argument('path', help='Path to code directory/file to commit')
    commit_parser.add_argument('-m', '--message', required=True, help='Commit message')
    commit_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    
    # Checkout command
    checkout_parser = subparsers.add_parser('checkout', help='Checkout code from the shared repository')
    checkout_parser.add_argument('path', help='Target path for checkout')
    checkout_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    checkout_parser.add_argument('-v', '--version', help='Specific version hash (optional, defaults to latest)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List versions in the shared repository')
    list_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    list_parser.add_argument('--filter-workspace', help='Optional: Filter versions by the workspace ID that committed them')
    
    # Pull command (replaces sync for shared repo)
    pull_parser = subparsers.add_parser('pull', help='Pull the latest code from the shared repository')
    pull_parser.add_argument('path', help='Target path for pulling the latest code')
    pull_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize version control with the current workspace's ID
    vc = S3VersionControl(args.bucket, args.passphrase, args.current_workspace_id)
    
    # Execute command
    if args.command == 'commit':
        vc.commit(args.path, args.message, args.branch)
    elif args.command == 'checkout':
        vc.checkout(args.path, args.branch, args.version)
    elif args.command == 'list':
        vc.list_versions(args.branch, args.filter_workspace)
    elif args.command == 'pull':
        vc.pull_latest(args.path, args.branch)

if __name__ == "__main__":
    main()
