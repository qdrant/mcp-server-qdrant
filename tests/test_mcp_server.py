import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_qdrant.mcp_server import (
    DefaultCollectionQdrantMCPServer,
    MultiCollectionQdrantMCPServer,
)
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)


@pytest.fixture
def embedding_settings():
    with patch.dict(
        os.environ,
        {
            "EMBEDDING_PROVIDER": "fastembed",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        },
    ):
        return EmbeddingProviderSettings()


@pytest.fixture
def tool_settings():
    with patch.dict(
        os.environ,
        {
            "TOOL_STORE_DESCRIPTION": "Store information",
            "TOOL_FIND_DESCRIPTION": "Find information",
        },
    ):
        return ToolSettings()


class TestDefaultCollectionQdrantMCPServer:
    QUERY_SCHEMA = {
        "title": "Query",
        "description": "A natural language query to search for.",
        "type": "string",
    }
    INFORMATION_SCHEMA = {
        "title": "Information",
        "description": "Natural language information to store.",
        "type": "string",
    }
    METADATA_SCHEMA = {
        "title": "Metadata",
        "description": "JSON metadata to store with the information, optional.",
        "type": "object",
        "default": None,
    }

    @pytest.mark.asyncio
    @patch("mcp_server_qdrant.mcp_server.create_embedding_provider")
    @patch("mcp_server_qdrant.mcp_server.QdrantConnector")
    async def test_init_with_default_collection(
        self,
        mock_qdrant_connector,
        mock_create_provider,
        embedding_settings,
        tool_settings,
    ):
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://localhost:6333",
                "COLLECTION_NAME": "test_collection",
                "QDRANT_READ_ONLY": "false",
            },
        ):
            qdrant_settings = QdrantSettings()

            mock_provider = MagicMock()
            mock_create_provider.return_value = mock_provider
            mock_connector = AsyncMock()
            mock_qdrant_connector.return_value = mock_connector

            server = DefaultCollectionQdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )

            registered_tools = await server.list_tools()
            tool_names = [tool.name for tool in registered_tools]

            assert "qdrant-find" in tool_names
            assert "qdrant-store" in tool_names
            assert len(tool_names) == 2

            # Verify find tool schema
            find_tool = next(
                tool for tool in registered_tools if tool.name == "qdrant-find"
            )
            assert find_tool.inputSchema["properties"]["query"] == self.QUERY_SCHEMA
            assert find_tool.inputSchema["required"] == ["query"]

            # Verify store tool schema
            store_tool = next(
                tool for tool in registered_tools if tool.name == "qdrant-store"
            )
            assert (
                store_tool.inputSchema["properties"]["information"]
                == self.INFORMATION_SCHEMA
            )
            assert (
                store_tool.inputSchema["properties"]["metadata"] == self.METADATA_SCHEMA
            )
            assert store_tool.inputSchema["required"] == ["information"]


class TestMultiCollectionQdrantMCPServer:
    QUERY_SCHEMA = {
        "title": "Query",
        "description": "A natural language query to search for.",
        "type": "string",
    }
    COLLECTION_NAME_FIND_SCHEMA = {
        "title": "Collection Name",
        "description": "Name of the collection to search in. Please list the existing collections to know which one to use.",
        "type": "string",
    }
    COLLECTION_NAME_STORE_SCHEMA = {
        "title": "Collection Name",
        "description": "Name of the collection to store the information in. Please list the existing collections to know which one to use.",
        "type": "string",
    }
    COLLECTION_NAME_CREATE_COLLECTION_SCHEMA = {
        "title": "Collection Name",
        "description": "Name of the collection to create.",
        "type": "string",
    }
    INFORMATION_SCHEMA = {
        "title": "Information",
        "description": "Natural language information to store.",
        "type": "string",
    }
    METADATA_SCHEMA = {
        "title": "Metadata",
        "description": "JSON metadata to store with the information, optional.",
        "type": "object",
        "default": None,
    }
    DESCRIPTION_SCHEMA = {
        "title": "Description",
        "description": "Purpose description of the collection.",
        "type": "string",
    }

    @pytest.mark.asyncio
    @patch("mcp_server_qdrant.mcp_server.create_embedding_provider")
    @patch("mcp_server_qdrant.mcp_server.QdrantConnector")
    async def test_init_multi_collection(
        self,
        mock_qdrant_connector,
        mock_create_provider,
        embedding_settings,
        tool_settings,
    ):
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://localhost:6333",
                "QDRANT_READ_ONLY": "false",
            },
        ):
            qdrant_settings = QdrantSettings()

            mock_provider = MagicMock()
            mock_create_provider.return_value = mock_provider
            mock_connector = AsyncMock()
            mock_qdrant_connector.return_value = mock_connector

            server = MultiCollectionQdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )

            registered_tools = await server.list_tools()
            tool_names = [tool.name for tool in registered_tools]

            assert "qdrant-find" in tool_names
            assert "qdrant-store" in tool_names
            assert "qdrant-list-collections" in tool_names
            assert "qdrant-create-collection" in tool_names
            assert len(tool_names) == 4

            # Verify find tool schema
            find_tool = next(
                tool for tool in registered_tools if tool.name == "qdrant-find"
            )
            assert find_tool.inputSchema["properties"]["query"] == self.QUERY_SCHEMA
            assert (
                find_tool.inputSchema["properties"]["collection_name"]
                == self.COLLECTION_NAME_FIND_SCHEMA
            )
            assert find_tool.inputSchema["required"] == ["query", "collection_name"]

            # Verify store tool schema
            store_tool = next(
                tool for tool in registered_tools if tool.name == "qdrant-store"
            )
            assert (
                store_tool.inputSchema["properties"]["information"]
                == self.INFORMATION_SCHEMA
            )
            assert (
                store_tool.inputSchema["properties"]["metadata"] == self.METADATA_SCHEMA
            )
            assert (
                store_tool.inputSchema["properties"]["collection_name"]
                == self.COLLECTION_NAME_STORE_SCHEMA
            )
            assert store_tool.inputSchema["required"] == [
                "information",
                "collection_name",
            ]

            # Verify list collections tool schema
            list_collections_tool = next(
                tool
                for tool in registered_tools
                if tool.name == "qdrant-list-collections"
            )
            assert list_collections_tool.inputSchema["properties"] == {}

            # Verify create collection tool schema
            create_collection_tool = next(
                tool
                for tool in registered_tools
                if tool.name == "qdrant-create-collection"
            )
            assert (
                create_collection_tool.inputSchema["properties"]["collection_name"]
                == self.COLLECTION_NAME_CREATE_COLLECTION_SCHEMA
            )
            assert (
                create_collection_tool.inputSchema["properties"]["description"]
                == self.DESCRIPTION_SCHEMA
            )
            assert create_collection_tool.inputSchema["required"] == [
                "collection_name",
                "description",
            ]

    @pytest.mark.asyncio
    @patch("mcp_server_qdrant.mcp_server.create_embedding_provider")
    @patch("mcp_server_qdrant.mcp_server.QdrantConnector")
    async def test_init_read_only_mode(
        self,
        mock_qdrant_connector,
        mock_create_provider,
        embedding_settings,
        tool_settings,
    ):
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://localhost:6333",
                "QDRANT_READ_ONLY": "true",
            },
        ):
            qdrant_settings = QdrantSettings()

            mock_provider = MagicMock()
            mock_create_provider.return_value = mock_provider
            mock_connector = AsyncMock()
            mock_qdrant_connector.return_value = mock_connector

            server = MultiCollectionQdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )

            registered_tools = await server.list_tools()
            tool_names = [tool.name for tool in registered_tools]

            assert "qdrant-find" in tool_names
            assert "qdrant-store" not in tool_names
            assert "qdrant-list-collections" in tool_names
            assert "qdrant-create-collection" not in tool_names
            assert len(tool_names) == 2

            # Verify find tool schema
            find_tool = next(
                tool for tool in registered_tools if tool.name == "qdrant-find"
            )
            assert find_tool.inputSchema["properties"]["query"] == self.QUERY_SCHEMA
            assert (
                find_tool.inputSchema["properties"]["collection_name"]
                == self.COLLECTION_NAME_FIND_SCHEMA
            )
            assert find_tool.inputSchema["required"] == ["query", "collection_name"]

            # Verify list collections tool schema
            list_collections_tool = next(
                tool
                for tool in registered_tools
                if tool.name == "qdrant-list-collections"
            )
            assert list_collections_tool.inputSchema["properties"] == {}


@pytest.mark.asyncio
@patch("mcp_server_qdrant.mcp_server.create_embedding_provider")
@patch("mcp_server_qdrant.mcp_server.QdrantConnector")
async def test_tool_descriptions(
    mock_qdrant_connector, mock_create_provider, embedding_settings
):
    with patch.dict(
        os.environ,
        {
            "TOOL_STORE_DESCRIPTION": "Custom store description",
            "TOOL_FIND_DESCRIPTION": "Custom find description",
            "QDRANT_URL": "http://localhost:6333",
            "COLLECTION_NAME": "test_collection",
            "QDRANT_READ_ONLY": "false",
        },
    ):
        custom_tool_settings = ToolSettings()
        qdrant_settings = QdrantSettings()

        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider
        mock_connector = AsyncMock()
        mock_qdrant_connector.return_value = mock_connector

        server = DefaultCollectionQdrantMCPServer(
            tool_settings=custom_tool_settings,
            qdrant_settings=qdrant_settings,
            embedding_provider_settings=embedding_settings,
        )

        registered_tools = await server.list_tools()
        find_tool = next(
            tool for tool in registered_tools if tool.name == "qdrant-find"
        )
        store_tool = next(
            tool for tool in registered_tools if tool.name == "qdrant-store"
        )

        assert find_tool.description == "Custom find description"
        assert store_tool.description == "Custom store description"
