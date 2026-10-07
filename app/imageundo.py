"""Short-lived, account-scoped recovery for a single studio image deletion."""
import copy
import hashlib
import json
import os
import re
import secrets
import shutil
import time

import pipeline

LOCK = pipeline.image_change_lock
FIELDS = ('versions', 'pic_info', 'version_info', 'keyframes_version', 'clips_version')


def snapshot(p):
    return {key: copy.deepcopy(p.get(key)) for key in FIELDS}


def fingerprint(path):
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def image_path(user, pid, rel):
    if pipeline.user_of(pid) != user or not re.fullmatch(r'keyframes/(?:kf_\d+\.png|history/kf_\d+_\d+\.png)', rel):
        raise ValueError('Image not found')
    root = os.path.realpath(pipeline.pdir(pid))
    path = os.path.realpath(pipeline.pdir(pid, *rel.split('/')))
    if not path.startswith(root + os.sep):
        raise ValueError('Image not found')
    return path


def idle(p):
    if pipeline.current['pid'] == p['id'] or re.match(r'^(queued|running)', p.get('status', '')):
        raise ValueError('Wait for generation to finish before changing images')


def delete(user, pid, rel):
    with LOCK:
        src = image_path(user, pid, rel)
        p = pipeline.load(pid)
        idle(p)
        if not os.path.isfile(src):
            raise ValueError('Image not found')
        before = snapshot(p)
        promoted = None
        if '/history/' in rel:
            name = rel.rsplit('/', 1)[1]
            for versions in p.get('versions', {}).values():
                if name in versions:
                    versions.remove(name)
            p.get('version_info', {}).pop(name, None)
        else:
            key = str(int(re.fullmatch(r'keyframes/kf_(\d+)\.png', rel).group(1)))
            versions = p.get('versions', {}).get(key, [])
            p.get('pic_info', {}).pop(key, None)
            if versions:
                name = versions.pop()
                promoted = 'keyframes/history/' + name
                image_path(user, pid, promoted)
                if name in p.get('version_info', {}):
                    p.setdefault('pic_info', {})[key] = p['version_info'].pop(name)
        promoted_path = image_path(user, pid, promoted) if promoted else None
        if promoted_path and not os.path.isfile(promoted_path):
            raise ValueError('Previous version is missing; image was kept')
        p['keyframes_version'] = p['clips_version'] = time.time()
        token = secrets.token_hex(24)
        folder = os.path.join(pipeline.TRASH, user, 'undo', token)
        os.makedirs(folder)
        trash = os.path.join(folder, 'deleted.png')
        record = dict(project=pid, path=rel, promoted=promoted, before=before, after=snapshot(p),
                      fingerprint=fingerprint(promoted_path) if promoted_path else None, expires=time.time() + 300)
        with open(os.path.join(folder, 'receipt.json'), 'w', encoding='utf-8') as stream:
            json.dump(record, stream)
        shutil.move(src, trash)
        try:
            if promoted_path:
                os.replace(promoted_path, src)
            pipeline.save(p)
        except Exception:
            if promoted_path and os.path.isfile(src):
                os.replace(src, promoted_path)
            shutil.move(trash, src)
            raise
        return {'deleted': 1, 'undo': token}


def restore(user, token):
    if not isinstance(token, str) or not re.fullmatch(r'[0-9a-f]{48}', token):
        raise ValueError('Undo is unavailable')
    with LOCK:
        folder = os.path.join(pipeline.TRASH, user, 'undo', token)
        receipt = os.path.join(folder, 'receipt.json')
        if not os.path.isfile(receipt):
            raise ValueError('Undo is unavailable')
        with open(receipt, encoding='utf-8') as stream:
            record = json.load(stream)
        if record['expires'] < time.time():
            raise ValueError('Undo expired; the image is still in Trash')
        pid = record['project']
        src = image_path(user, pid, record['path'])
        p = pipeline.load(pid)
        idle(p)
        trash = os.path.join(folder, 'deleted.png')
        promoted = image_path(user, pid, record['promoted']) if record['promoted'] else None
        if (snapshot(p) != record['after'] or fingerprint(src) != record['fingerprint']
                or not os.path.isfile(trash) or (promoted and os.path.exists(promoted))):
            raise ValueError('The project changed; newer images were kept. The deleted image is in Trash')
        if promoted:
            os.replace(src, promoted)
        try:
            shutil.move(trash, src)
        except Exception:
            if promoted:
                os.replace(promoted, src)
            raise
        try:
            for key, value in record['before'].items():
                if value is None:
                    p.pop(key, None)
                else:
                    p[key] = value
            p['keyframes_version'] = p['clips_version'] = time.time()
            pipeline.save(p)
        except Exception:
            shutil.move(src, trash)
            if promoted:
                os.replace(promoted, src)
            raise
        os.remove(receipt)
        return {'restored': 1, 'project': pid}
