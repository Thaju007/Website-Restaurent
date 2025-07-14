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
    def __init__(self, bucket_name, passphrase, workspace_id):
        """
        Initialize S3 Version Control
        
        Args:
            bucket_name: S3 bucket name for storing code
            passphrase: Your secret passphrase for encryption
            workspace_id: Unique identifier for your workspace (e.g., 'workspace1', 'workspace2')
        """
        self.bucket_name = bucket_name
        self.workspace_id = workspace_id
        self.s3_client = boto3.client('s3')
        
        # Generate encryption key from passphrase
        self.cipher_suite = self._generate_cipher(passphrase)
        
        # Base paths in S3
        self.base_path = f"code-repo/{workspace_id}"
        self.versions_path = f"{self.base_path}/versions"
        self.metadata_path = f"{self.base_path}/metadata"
        
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
        Commit code to S3 repository
        
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
            
            # Create metadata
            metadata = {
                'workspace_id': self.workspace_id,
                'timestamp': timestamp,
                'commit_message': commit_message,
                'branch': branch,
                'hash': version_hash,
                'size': len(archive_data)
            }
            
            # Encrypt archive
            encrypted_archive = self._encrypt_data(archive_data)
            
            # Upload encrypted archive
            version_key = f"{self.versions_path}/{branch}/{version_hash}.zip.enc"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=version_key,
                Body=encrypted_archive
            )
            
            # Upload metadata
            metadata_key = f"{self.metadata_path}/{branch}/{version_hash}.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2)
            )
            
            # Update branch pointer
            branch_key = f"{self.base_path}/branches/{branch}.json"
            branch_info = {
                'latest_hash': version_hash,
                'latest_timestamp': timestamp,
                'workspace_id': self.workspace_id
            }
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=branch_key,
                Body=json.dumps(branch_info, indent=2)
            )
            
            print(f"✅ Committed successfully!")
            print(f"   Version: {version_hash[:8]}")
            print(f"   Branch: {branch}")
            print(f"   Message: {commit_message}")
            print(f"   Timestamp: {timestamp}")
            
            return version_hash
            
        except Exception as e:
            print(f"❌ Error during commit: {e}")
            return None
    
    def checkout(self, target_path, branch="main", version_hash=None):
        """
        Checkout code from S3 repository
        
        Args:
            target_path: Path where to extract code
            branch: Branch to checkout (default: main)
            version_hash: Specific version hash (optional, defaults to latest)
        """
        try:
            # Get version hash if not provided
            if not version_hash:
                version_hash = self._get_latest_version(branch)
                if not version_hash:
                    print(f"❌ No versions found for branch '{branch}'")
                    return False
            
            # Download encrypted archive
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
            
            print(f"✅ Checkout successful!")
            print(f"   Version: {version_hash[:8]}")
            print(f"   Branch: {branch}")
            print(f"   Target: {target_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during checkout: {e}")
            return False
    
    def _get_latest_version(self, branch):
        """Get latest version hash for a branch"""
        try:
            branch_key = f"{self.base_path}/branches/{branch}.json"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=branch_key)
            branch_info = json.loads(response['Body'].read())
            return branch_info['latest_hash']
        except:
            return None
    
    def list_versions(self, branch="main"):
        """List all versions in a branch"""
        try:
            prefix = f"{self.metadata_path}/{branch}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                print(f"No versions found for branch '{branch}'")
                return []
            
            versions = []
            for obj in response['Contents']:
                # Get metadata
                metadata_response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=obj['Key']
                )
                metadata = json.loads(metadata_response['Body'].read())
                versions.append(metadata)
            
            # Sort by timestamp (newest first)
            versions.sort(key=lambda x: x['timestamp'], reverse=True)
            
            print(f"📋 Version History for branch '{branch}':")
            print("-" * 80)
            for version in versions:
                print(f"Hash: {version['hash'][:8]}")
                print(f"Date: {version['timestamp']}")
                print(f"Message: {version['commit_message']}")
                print(f"Workspace: {version['workspace_id']}")
                print("-" * 40)
            
            return versions
            
        except Exception as e:
            print(f"❌ Error listing versions: {e}")
            return []
    
    def sync_from_workspace(self, other_workspace_id, branch="main"):
        """
        Sync code from another workspace
        
        Args:
            other_workspace_id: ID of the workspace to sync from
            branch: Branch to sync
        """
        try:
            # Get latest version from other workspace
            other_branch_key = f"code-repo/{other_workspace_id}/branches/{branch}.json"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=other_branch_key)
            other_branch_info = json.loads(response['Body'].read())
            
            version_hash = other_branch_info['latest_hash']
            
            # Copy version files to current workspace
            source_version_key = f"code-repo/{other_workspace_id}/versions/{branch}/{version_hash}.zip.enc"
            target_version_key = f"{self.versions_path}/{branch}/{version_hash}.zip.enc"
            
            # Copy encrypted archive
            self.s3_client.copy_object(
                Bucket=self.bucket_name,
                CopySource={'Bucket': self.bucket_name, 'Key': source_version_key},
                Key=target_version_key
            )
            
            # Copy metadata
            source_metadata_key = f"code-repo/{other_workspace_id}/metadata/{branch}/{version_hash}.json"
            target_metadata_key = f"{self.metadata_path}/{branch}/{version_hash}.json"
            
            self.s3_client.copy_object(
                Bucket=self.bucket_name,
                CopySource={'Bucket': self.bucket_name, 'Key': source_metadata_key},
                Key=target_metadata_key
            )
            
            # Update branch pointer
            branch_key = f"{self.base_path}/branches/{branch}.json"
            branch_info = {
                'latest_hash': version_hash,
                'latest_timestamp': other_branch_info['latest_timestamp'],
                'workspace_id': self.workspace_id,
                'synced_from': other_workspace_id
            }
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=branch_key,
                Body=json.dumps(branch_info, indent=2)
            )
            
            print(f"✅ Synced from workspace '{other_workspace_id}'!")
            print(f"   Version: {version_hash[:8]}")
            print(f"   Branch: {branch}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during sync: {e}")
            return False

# Usage examples and CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='S3 Version Control System')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--workspace', required=True, help='Workspace ID')
    parser.add_argument('--passphrase', required=True, help='Encryption passphrase')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Commit command
    commit_parser = subparsers.add_parser('commit', help='Commit code')
    commit_parser.add_argument('path', help='Path to code directory/file')
    commit_parser.add_argument('-m', '--message', required=True, help='Commit message')
    commit_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    
    # Checkout command
    checkout_parser = subparsers.add_parser('checkout', help='Checkout code')
    checkout_parser.add_argument('path', help='Target path for checkout')
    checkout_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    checkout_parser.add_argument('-v', '--version', help='Specific version hash')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List versions')
    list_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    
    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Sync from another workspace')
    sync_parser.add_argument('workspace', help='Source workspace ID')
    sync_parser.add_argument('-b', '--branch', default='main', help='Branch name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize version control
    vc = S3VersionControl(args.bucket, args.passphrase, args.workspace)
    
    # Execute command
    if args.command == 'commit':
        vc.commit(args.path, args.message, args.branch)
    elif args.command == 'checkout':
        vc.checkout(args.path, args.branch, args.version)
    elif args.command == 'list':
        vc.list_versions(args.branch)
    elif args.command == 'sync':
        vc.sync_from_workspace(args.workspace, args.branch)

if __name__ == "__main__":
    main()
