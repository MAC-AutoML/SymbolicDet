import json
from dataclasses import dataclass
from typing import Any, Dict, List

class MessageType:
    INIT = "init"
    EVOLUTION_UPDATE = "evolution_update"
    THRESHOLD_START = "threshold_start"
    SUGGESTION = "suggestion"
    ERROR = "error"
    COMMAND = "command" 

@dataclass
class Suggestion:
    expression: str
    reason: str

@dataclass
class Message:
    msg_type: str
    payload: Dict[str, Any]
    
    def serialize(self) -> str:
        return json.dumps({
            "type": self.msg_type,
            "payload": self.payload
        })
    
    @classmethod
    def deserialize(cls, data: str) -> 'Message':
        obj = json.loads(data)
        return cls(
            msg_type=obj['type'],
            payload=obj['payload']
        )
    
    def get_suggestions(self) -> List[Suggestion]:
        if self.msg_type != MessageType.SUGGESTION:
            raise ValueError("Invalid message type for suggestions")
        return [Suggestion(**s) for s in self.payload.get('suggestions', [])] 