#!/usr/bin/env python3
"""
S3-Based Version Control System for EMR Workspaces
A lightweight version control system using S3 with encryption.
- Allows specifying a repository root prefix within the S3 bucket.
- Supports committing from local paths or S3 URIs.
- Does NOT track specific workspace IDs for commits.
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
from urllib.parse import urlparse

class S3VersionControl:
    def __init__(self, bucket_name, passphrase, repository_root_prefix="emr-shared-code-repo"):
        """
        Initialize S3 Version Control
        
        Args:
            bucket_name: S3 bucket name for storing code (the target repository bucket)
            passphrase: Your secret passphrase for encryption
            repository_root_prefix: The base prefix within the S3 bucket where this specific
                                    version control repository's data will be stored.
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        
        # Ensure repository_root_prefix does not start or end with a '/'
        self.repository_root_prefix = repository_root_prefix.strip('/')
        if not self.repository_root_prefix:
            self.repository_root_prefix = "emr-shared-code-repo" # Fallback to default
        
        # Base paths in S3 for THIS specific repository instance
        self.versions_path = f"{self.repository_root_prefix}/versions"
        self.metadata_path = f"{self.repository_root_prefix}/metadata"
        self.branches_path = f"{self.repository_root_prefix}/branches" # Pointer for branches

        # Generate encryption key from passphrase
        self.cipher_suite = self._generate_cipher(passphrase)

    def _generate_cipher(self, passphrase):
        """Generate Fernet cipher from passphrase"""
        password = passphrase.encode()
        # In production, for multiple clients sharing the same repo,
        # the salt MUST be identical and securely known by all clients.
        # For simplicity, using a fixed salt here.
        salt = b'salt_for_consistency'
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
    
    def _create_archive_from_local(self, source_path):
        """Create zip archive of source directory from local file system"""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            source_path = Path(source_path)
            
            if source_path.is_file():
                zip_file.write(source_path, source_path.name)
            else: # If source_path is a directory
                for file_path in source_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(source_path)
                        zip_file.write(file_path, relative_path)
        
        buffer.seek(0)
        return buffer.getvalue()

    def _create_archive_from_s3(self, s3_source_uri: str) -> bytes:
        """
        Creates an in-memory zip archive from an S3 directory (prefix) or a single S3 file.
        s3_source_uri should be like "s3://my-bucket/my-folder/" or "s3://my-bucket/my-file.txt"
        """
        parsed_uri = urlparse(s3_source_uri)
        source_bucket = parsed_uri.netloc
        source_key_prefix = parsed_uri.path.lstrip('/')

        if not source_bucket:
            raise ValueError(f"Invalid S3 URI: '{s3_source_uri}'. Missing bucket name.")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Check if it's a single file or a directory (prefix)
            if not source_key_prefix.endswith('/') and '.' in Path(source_key_prefix).name:
                # Treat as a single file
                try:
                    obj_data = self.s3_client.get_object(Bucket=source_bucket, Key=source_key_prefix)['Body'].read()
                    zip_file.writestr(Path(source_key_prefix).name, obj_data)
                    print(f"Archived single S3 file: s3://{source_bucket}/{source_key_prefix}")
                except self.s3_client.exceptions.NoSuchKey:
                    raise FileNotFoundError(f"S3 object not found: s3://{source_bucket}/{source_key_prefix}")
            else:
                # Treat as a directory (prefix)
                if source_key_prefix and not source_key_prefix.endswith('/'):
                    source_key_prefix += '/'

                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=source_bucket, Prefix=source_key_prefix)
                
                found_files = 0
                for page in pages:
                    if "Contents" in page:
                        for obj in page['Contents']:
                            object_key = obj['Key']
                            
                            if object_key == source_key_prefix or object_key.endswith('/'):
                                continue

                            relative_path = Path(object_key).relative_to(Path(source_key_prefix))
                            
                            try:
                                obj_data = self.s3_client.get_object(Bucket=source_bucket, Key=object_key)['Body'].read()
                                zip_file.writestr(str(relative_path), obj_data)
                                found_files += 1
                            except self.s3_client.exceptions.NoSuchKey:
                                print(f"Warning: S3 object s3://{source_bucket}/{object_key} disappeared during archiving. Skipping.")
                
                if found_files == 0:
                    raise FileNotFoundError(f"No files found under S3 prefix: s3://{source_bucket}/{source_key_prefix}")
                
                print(f"Archived {found_files} files from S3 prefix: s3://{source_bucket}/{source_key_prefix}")

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    def commit(self, source_path, commit_message, branch="main"):
        """
        Commit code to the shared S3 repository
        
        Args:
            source_path: Path to code directory/file (local) or S3 URI (s3://bucket/prefix/)
            commit_message: Commit message
            branch: Branch name (default: main)
        """
        try:
            archive_data = None
            if source_path.startswith("s3://"):
                print(f"Committing from S3 location: {source_path}")
                archive_data = self._create_archive_from_s3(source_path)
            else:
                print(f"Committing from local path: {source_path}")
                archive_data = self._create_archive_from_local(source_path)
            
            if archive_data is None or len(archive_data) == 0:
                print(f"❌ Error: No data archived from {source_path}. Aborting commit.")
                return None

            timestamp = datetime.datetime.now().isoformat()
            version_hash = self._calculate_hash(archive_data)
            
            # Metadata no longer includes 'committer_workspace_id'
            metadata = {
                'timestamp': timestamp,
                'commit_message': commit_message,
                'branch': branch,
                'hash': version_hash,
                'size': len(archive_data),
                'source_type': 's3' if source_path.startswith("s3://") else 'local',
                'original_source_path': source_path
            }
            
            encrypted_archive = self._encrypt_data(archive_data)
            
            version_key = f"{self.versions_path}/{branch}/{version_hash}.zip.enc"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=version_key,
                Body=encrypted_archive
            )
            
            metadata_key = f"{self.metadata_path}/{branch}/{version_hash}.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2)
            )
            
            branch_key = f"{self.branches_path}/{branch}.json"
            # Branch info no longer includes 'last_committed_by_workspace_id'
            branch_info = {
                'latest_hash': version_hash,
                'latest_timestamp': timestamp,
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
            print(f"    Source: {source_path}")
            print(f"    Repository Root: s3://{self.bucket_name}/{self.repository_root_prefix}/")
            
            return version_hash
            
        except FileNotFoundError as e:
            print(f"❌ Error: Source not found: {e}")
            return None
        except ValueError as e:
            print(f"❌ Configuration Error: {e}")
            return None
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
            if not version_hash:
                version_hash = self._get_latest_version(branch)
                if not version_hash:
                    print(f"❌ No versions found for branch '{branch}' in the shared repository located at s3://{self.bucket_name}/{self.repository_root_prefix}/")
                    return False
            
            version_key = f"{self.versions_path}/{branch}/{version_hash}.zip.enc"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=version_key)
            encrypted_data = response['Body'].read()
            
            decrypted_data = self._decrypt_data(encrypted_data)
            
            target_path = Path(target_path)
            target_path.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(io.BytesIO(decrypted_data), 'r') as zip_file:
                zip_file.extractall(target_path)
            
            print(f"✅ Checkout successful from shared repository!")
            print(f"    Version: {version_hash[:8]}")
            print(f"    Branch: {branch}")
            print(f"    Target: {target_path}")
            print(f"    Repository Root: s3://{self.bucket_name}/{self.repository_root_prefix}/")
            
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
        except self.s3_client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            print(f"Warning: Could not retrieve latest version for branch '{branch}' from s3://{self.bucket_name}/{self.repository_root_prefix}/. {e}")
            return None
    
    def list_versions(self, branch="main"):
        """
        List all versions in a branch from the shared repository.
        The filtering by workspace ID is removed.
        """
        try:
            prefix = f"{self.metadata_path}/{branch}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                print(f"No versions found for branch '{branch}' in the shared repository located at s3://{self.bucket_name}/{self.repository_root_prefix}/.")
                return []
            
            versions = []
            for obj in response['Contents']:
                metadata_response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=obj['Key']
                )
                metadata = json.loads(metadata_response['Body'].read())
                versions.append(metadata)
            
            versions.sort(key=lambda x: x['timestamp'], reverse=True)
            
            print(f"📋 Version History for branch '{branch}' (Shared Repository at s3://{self.bucket_name}/{self.repository_root_prefix}/):")
            print("-" * 80)
            for version in versions:
                print(f"Hash: {version['hash'][:8]}")
                print(f"Date: {version['timestamp']}")
                print(f"Message: {version['commit_message']}")
                # print(f"Committed by Workspace: {version.get('committer_workspace_id', 'N/A')}") # Removed
                print(f"Source Type: {version.get('source_type', 'N/A')}")
                print(f"Original Source: {version.get('original_source_path', 'N/A')[:60]}...")
                print("-" * 40)
            
            return versions
            
        except Exception as e:
            print(f"❌ Error listing versions: {e}")
            return []
    
    def pull_latest(self, target_path, branch="main"):
        """
        Pulls the latest version from the shared repository for a given branch.
        """
        print(f"Attempting to pull latest for branch '{branch}' to '{target_path}' from s3://{self.bucket_name}/{self.repository_root_prefix}/...")
        return self.checkout(target_path, branch)

# Usage examples and CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='S3 Version Control System for Shared EMR Workspaces')
    parser.add_argument('--bucket', required=True, help='S3 bucket name (the main bucket)')
    parser.add_argument('--passphrase', required=True, help='Encryption passphrase')
    parser.add_argument('--repo-prefix', default='emr-shared-code-repo', 
                        help='Optional: The prefix/folder inside the S3 bucket where this VCS instance resides (default: emr-shared-code-repo)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Commit command
    commit_parser = subparsers.add_parser('commit', help='Commit code to the shared repository (can be local path or s3:// URI)')
    commit_parser.add_argument('path', help='Path to code directory/file (local) OR S3 URI (e.g., s3://my-source-bucket/my-folder/)')
    commit_parser.add_argument('-m', '--message', required=True, help='Commit message')
    commit_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    
    # Checkout command
    checkout_parser = subparsers.add_parser('checkout', help='Checkout code from the shared repository')
    checkout_parser.add_argument('path', help='Target local path for checkout')
    checkout_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    checkout_parser.add_argument('-v', '--version', help='Specific version hash')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List versions in the shared repository')
    list_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    # Filter by workspace is removed
    
    # Pull command
    pull_parser = subparsers.add_parser('pull', help='Pull the latest code from the shared repository')
    pull_parser.add_argument('path', help='Target local path for pulling the latest code')
    pull_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize version control without current_workspace_id
    vc = S3VersionControl(args.bucket, args.passphrase, args.repo_prefix)
    
    # Execute command
    if args.command == 'commit':
        vc.commit(args.path, args.message, args.branch)
    elif args.command == 'checkout':
        vc.checkout
