#!/usr/bin/env python3

import logging
import json
import tqdm
from splitstream import splitfile

logger = logging.getLogger('analyze_alignment_history')

def load_and_organize(filename: str, max_merge_interval=4: int):
    """load_and_organize

    Loads the alignment history file from `filename`, parses the JSONs
    and organizes them so that each "session" is a separate entry in a
    list.

    A "session" is determined as follows:
    1. An "align" (i.e. manual alignment) entry starts a new session
    2. A "load" (of a pre-existing alignment) starts a new session if
    the time since the previous entry (align or sync or load) is less
    than `max_merge_interval` in hours.
    3. A "sync" never starts a new session and is appended to the
    existing session

    Note: Never set the `max_merge_interval` to a time longer than about
    14 hours (a typical session length) because then multiple nights'
    alignments may get transitively concatenated. This prevents abuse of
    "sync" to realign from a previous night from being merged.

    Returns a list of lists of dictionaries. Each outer list entry is a
    session.
    """

    jsonstream = []
    with open(filename, 'r') as f:
        for idx, jsonitem in tqdm.tqdm(enumerate(splitfile(f, format='json'))):
            try:
                data = json.loads(jsonitem)
            except:
                logger.error(f'Encountered exception while analyzing JSON stub #{idx} (0-based) in the file {filename}. Please investigate and fix the entry manually and then try again!')
                raise
            jsonstream.append(data)

    # Easy cases:
    if len(jsonstream) == 0:
        return []
    if len(jsonstream) == 1:
        return [jsonstream,]

    sessions = [] if jsonstream[0]['source'] == 'align' else [[]]
    previous = None
    t0 = -1 # Invalid UNIX time
    for idx, data in tqdm.tqdm(enumerate(jsonstream)):
        data = dict(data, index=idx) # Index in this file, useful for debugging since it corresponds to line number
        if data['source'] == 'align':
            sessions.append([])
            sessions[-1].append(data)
        elif data['source'] == 'loaded':
            if (data['timestamp'] - t0)/3600.0 > max_merge_interval:
                sessions.append([]) # New session
            sessions[-1].append(data)
        elif data['source'] == 'sync':
            if len(sessions[-1]) == 0:
                assert idx == 0, f'Programming error: {idx}'
            else:
                logger.error(f'File {filename} starts with a `sync`!')
            sessions[-1].append(data)

    logger.info(f'{len(sessions)} sessions loaded.')
    return sessions


def munge(sessions):
    """munge

    Given a list of sessions, munge the JSONs as follows:

    * If the session is length 1, we inform the user and ignore it.

    * If the session starts with a sync, it's a bug; alert the user and
      discard the session.

    * If the session starts with a load, it's a non-trustworthy session;
      alert the user and discard the session

    * If the session starts with an align, then we process the session
      as described below

    Session processing:

    1. We note down the details of the starting alignment, which is the
       only alignment in the session. Notably, we log the real-sky
       alt/az of the alignment target.

    FIXME: Complete
    """

    raise NotImplementedError('TODO')
