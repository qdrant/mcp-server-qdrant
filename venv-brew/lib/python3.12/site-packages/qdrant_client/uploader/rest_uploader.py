from itertools import count
from time import sleep
from typing import Any, Iterable, Optional, Union
from uuid import uuid4

import numpy as np

from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http import SyncApis
from qdrant_client.http.models import Batch, PointsList, PointStruct, ShardKeySelector
from qdrant_client.uploader.uploader import BaseUploader
from qdrant_client.common.client_warnings import show_warning


def upload_batch(
    openapi_client: SyncApis,
    collection_name: str,
    batch: Union[tuple, Batch],
    max_retries: int,
    shard_key_selector: Optional[ShardKeySelector],
    wait: bool = False,
) -> bool:
    ids_batch, vectors_batch, payload_batch = batch

    ids_batch = (str(uuid4()) for _ in count()) if ids_batch is None else ids_batch
    payload_batch = (None for _ in count()) if payload_batch is None else payload_batch

    points = [
        PointStruct(
            id=idx,
            vector=(vector.tolist() if isinstance(vector, np.ndarray) else vector) or {},
            payload=payload,
        )
        for idx, vector, payload in zip(ids_batch, vectors_batch, payload_batch)
    ]

    attempt = 0
    while attempt < max_retries:
        try:
            openapi_client.points_api.upsert_points(
                collection_name=collection_name,
                point_insert_operations=PointsList(points=points, shard_key=shard_key_selector),
                wait=wait,
            )
            break
        except ResourceExhaustedResponse as ex:
            show_warning(
                message=f"Batch upload failed due to rate limit. Waiting for {ex.retry_after_s} seconds before retrying...",
                category=UserWarning,
                stacklevel=7,
            )
            sleep(ex.retry_after_s)

        except Exception as e:
            show_warning(
                message=f"Batch upload failed {attempt + 1} times. Retrying...",
                category=UserWarning,
                stacklevel=7,
            )

            if attempt == max_retries - 1:
                raise e

            attempt += 1
    return True


class RestBatchUploader(BaseUploader):
    def __init__(
        self,
        uri: str,
        collection_name: str,
        max_retries: int,
        wait: bool = False,
        shard_key_selector: Optional[ShardKeySelector] = None,
        **kwargs: Any,
    ):
        self.collection_name = collection_name
        self.openapi_client: SyncApis = SyncApis(host=uri, **kwargs)
        self.max_retries = max_retries
        self._wait = wait
        self._shard_key_selector = shard_key_selector

    @classmethod
    def start(
        cls,
        collection_name: Optional[str] = None,
        uri: str = "http://localhost:6333",
        max_retries: int = 3,
        **kwargs: Any,
    ) -> "RestBatchUploader":
        if not collection_name:
            raise RuntimeError("Collection name could not be empty")
        return cls(uri=uri, collection_name=collection_name, max_retries=max_retries, **kwargs)

    def process(self, items: Iterable[Any]) -> Iterable[bool]:
        for batch in items:
            yield upload_batch(
                self.openapi_client,
                self.collection_name,
                batch,
                shard_key_selector=self._shard_key_selector,
                max_retries=self.max_retries,
                wait=self._wait,
            )
