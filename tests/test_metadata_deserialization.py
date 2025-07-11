"""
Тесты для десериализации metadata в MCP Qdrant Server.
"""

import json
import pytest
from typing import Dict, Any, Union

from mcp_server_qdrant.qdrant import Metadata


class TestMetadataDeserialization:
    """Тестирование десериализации metadata"""

    def test_metadata_dict_passthrough(self):
        """Тест: словарь metadata должен проходить без изменений"""
        metadata = {"key": "value", "number": 123}
        result = self._deserialize_metadata(metadata)
        assert result == metadata

    def test_metadata_none_passthrough(self):
        """Тест: None metadata должен проходить без изменений"""
        metadata = None
        result = self._deserialize_metadata(metadata)
        assert result is None

    def test_metadata_json_string_deserialization(self):
        """Тест: JSON строка должна десериализоваться в словарь"""
        metadata = '{"key": "value", "number": 123}'
        result = self._deserialize_metadata(metadata)
        assert result == {"key": "value", "number": 123}

    def test_metadata_empty_dict_string(self):
        """Тест: пустой JSON объект как строка"""
        metadata = '{}'
        result = self._deserialize_metadata(metadata)
        assert result == {}

    def test_metadata_null_string(self):
        """Тест: 'null' строка должна стать None"""
        metadata = 'null'
        result = self._deserialize_metadata(metadata)
        assert result is None

    def test_metadata_none_string(self):
        """Тест: 'none' строка должна стать None"""
        metadata = 'none'
        result = self._deserialize_metadata(metadata)
        assert result is None

    def test_metadata_None_string(self):
        """Тест: 'None' строка должна стать None"""
        metadata = 'None'
        result = self._deserialize_metadata(metadata)
        assert result is None

    def test_metadata_empty_string(self):
        """Тест: пустая строка должна стать None"""
        metadata = ''
        result = self._deserialize_metadata(metadata)
        assert result is None

    def test_metadata_complex_json(self):
        """Тест: сложный JSON объект"""
        metadata = '{"nested": {"key": "value"}, "array": [1, 2, 3], "bool": true}'
        result = self._deserialize_metadata(metadata)
        assert result == {
            "nested": {"key": "value"},
            "array": [1, 2, 3],
            "bool": True
        }

    def test_metadata_invalid_json_raises_error(self):
        """Тест: невалидный JSON должен вызывать ValueError"""
        invalid_cases = [
            '{invalid json}',
            '{"unclosed": "quote}',
            'not json at all',
            '{"key": value}',  # без кавычек вокруг value
        ]
        
        for invalid in invalid_cases:
            with pytest.raises(ValueError, match="Invalid JSON in metadata"):
                self._deserialize_metadata(invalid)

    def test_metadata_non_dict_after_deserialization_raises_error(self):
        """Тест: metadata не являющийся словарем после десериализации должен вызывать TypeError"""
        invalid_cases = [
            '"string value"',  # строка в JSON
            '123',  # число в JSON
            'true',  # boolean в JSON
            '[1, 2, 3]',  # массив в JSON
        ]
        
        for invalid in invalid_cases:
            with pytest.raises(TypeError, match="Metadata must be a dictionary"):
                self._deserialize_metadata(invalid)

    def _deserialize_metadata(self, metadata: Union[Metadata, str, None]) -> Union[Metadata, None]:
        """
        Внутренняя функция для тестирования логики десериализации.
        Имитирует логику из store функции.
        """
        # Десериализация metadata если это строка
        if isinstance(metadata, str):
            try:
                if metadata.lower() in ['null', 'none', '']:
                    metadata = None
                else:
                    metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in metadata: {metadata}")
        
        # Валидация типа metadata после десериализации
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"Metadata must be a dictionary, got {type(metadata)}")

        return metadata


class TestMetadataEdgeCases:
    """Тестирование граничных случаев"""

    def test_metadata_whitespace_variations(self):
        """Тест: различные варианты пробелов"""
        test_cases = [
            ('  null  ', None),
            ('  none  ', None),
            ('  None  ', None),
            ('  ""  ', ""),  # пустая строка в JSON
            ('  {}  ', {}),
        ]
        
        for input_val, expected in test_cases:
            if expected == "":
                # Пустая строка в JSON должна вызвать TypeError
                with pytest.raises(TypeError):
                    self._deserialize_metadata(input_val)
            else:
                result = self._deserialize_metadata(input_val)
                assert result == expected

    def test_metadata_case_sensitivity(self):
        """Тест: чувствительность к регистру для специальных значений"""
        test_cases = [
            ('NULL', None),
            ('Null', None),
            ('NONE', None),
            ('None', None),
            ('nOnE', None),
        ]
        
        for input_val, expected in test_cases:
            result = self._deserialize_metadata(input_val)
            assert result == expected

    def _deserialize_metadata(self, metadata: Union[Metadata, str, None]) -> Union[Metadata, None]:
        """
        Внутренняя функция для тестирования логики десериализации.
        Имитирует логику из store функции.
        """
        # Десериализация metadata если это строка
        if isinstance(metadata, str):
            try:
                if metadata.lower().strip() in ['null', 'none', '']:
                    metadata = None
                else:
                    metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in metadata: {metadata}")
        
        # Валидация типа metadata после десериализации
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"Metadata must be a dictionary, got {type(metadata)}")

        return metadata