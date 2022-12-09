import gzip
import importlib
import os
import pickle
import sys

from colorama import Fore

from general.conf import settings


def log(message):
    if settings.DEBUG:
        message = message.replace("\n", "\n    ")
        sys.stderr.write(Fore.YELLOW + (" >> %s\n" % message) + Fore.RESET)


def make_sure_dir_exists(dire_name):
    import errno

    try:
        os.makedirs(dire_name)
        log("Dir `%s` created successfully." % dire_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def calculate_md5_sum(file_path):
    import hashlib

    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url, file_path, md5_sum=None):
    from six.moves.urllib.request import urlretrieve

    log("Downloading `%s` ..." % url)
    urlretrieve(url, filename=file_path)
    log("Downloaded to `%s` ..." % file_path)

    if md5_sum and calculate_md5_sum(file_path) != md5_sum:
        raise Exception("Hash mismatch (file downloaded from `%s`)" % url)


def download_file_if_not_exists(url, file_path, md5_sum):
    if os.path.exists(file_path) and calculate_md5_sum(file_path) == md5_sum:
        log("File `%s` already exists." % file_path)
        return
    download_file(url, file_path, md5_sum=md5_sum)


def _extract_compressed_file(source, destination):
    if source.endswith(".zip"):
        from zipfile import ZipFile
        with ZipFile(source, 'r') as zip_file:
            zip_file.extractall(destination)
            log("Extracted `%s` to \n"
                "          `%s`" % (source, destination))
    elif source.endswith(".tar.gz"):
        import tarfile
        with tarfile.open(source, "r:gz") as tar_file:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_file, path=destination)
            log("Extracted `%s` to \n"
                "          `%s`" % (source, destination))
    elif source.endswith(".tar"):
        import tarfile
        with tarfile.open(source, "r") as tar_file:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_file, path=destination)
            log("Extracted `%s` to \n"
                "          `%s`" % (source, destination))
    else:
        raise NotImplementedError


def extract_compressed_file(source, destination, override=False):
    status_file_path = os.path.join(destination, ".status")

    # Skip if file is extracted successfully
    if not override and os.path.exists(status_file_path):
        with open(status_file_path, 'r') as extraction_status_file:
            extraction_status_content = extraction_status_file.read()
            if "EXTRACTION STATUS: SUCCESSFUL" in extraction_status_content and \
                    "SOURCE FILE: `%s`" % os.path.basename(source) in extraction_status_content:
                log("File `%s` is already extracted." % source)
                return

    # Here we extract file if needed
    _extract_compressed_file(source, destination)
    with open(status_file_path, 'w') as extraction_status_file:
        extraction_status_file.write("EXTRACTION STATUS: SUCCESSFUL\n")
        extraction_status_file.write("SOURCE FILE: `%s`\n" % os.path.basename(source))


def generate_seed():
    import time

    seed = int(time.time() * 1e5) % 2 ** 23
    log("Seed `%s` generated." % seed)

    return seed


def load_class(class_path):
    module_name = ".".join(class_path.split(".")[:-1])
    class_name = class_path.split(".")[-1]
    module = importlib.import_module(module_name)
    loaded_class = getattr(module, class_name)
    return loaded_class


def dump_gzip_pickle(obj, path):
    with gzip.open(path, 'wb') as file:
        pickle.dump(obj, file)


def load_gzip_pickle(path):
    with gzip.open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj
