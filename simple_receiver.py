""" Test script for a receiver which listens for instructions from Azure.

Upon receiving a simulation instructions a proxy for a reservoir simulation
is called, and the results are sent back to Azure.

USAGE:
    0) Store the passwords for access to Azure in a file config.json, with the format
     {
     "access_send" = password_send,
     "access_receive" = password_receive
     }
    1) Start this script and let it run (forever)
    2) Send instructions, e.g. by the script simple_sender.py

"""
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict
import numpy as np

# pip install azure-iot-device
from azure.iot.device import IoTHubDeviceClient, Message

# pip install azure-eventhub
from azure.eventhub import EventHubConsumerClient, EventData

# pip install azure-storage-blob
from azure.storage.blob import BlobClient


class ResultSender:
    """Send information that can be interpreted as instructions for a simulation"""

    def __init__(self, client):
        self._client = client

    def simulation_data(self, instructions: List):
        # Call runscript for simulator here?
        return {"pressure": instructions + instructions}

    def make_message(self, instruction):
        return {
            "type": "simulation_result",
            "time": datetime.now(timezone.utc).isoformat(),
            "values": self.simulation_data(instruction),
        }

    def send_data(self, instruction):
        data = self.make_message(instruction)
        message = Message(json.dumps(data))
        self._client.send_message(message)
        print("Simulation data sent at ", data["time"])


with open("config.json", "r") as f:
    config = json.load(f)

client_send = IoTHubDeviceClient.create_from_connection_string(config["access_send"])
sender = ResultSender(client_send)

# This function is called when new events are received,
# or when a sufficient time has passed without any events
def on_events(particion_context: int, events: List[EventData]):
    now = datetime.now(timezone.utc).isoformat()
    for event in events:
        data = event.body_as_json()
        msg_type = data["type"]
        sent_time = data["time"]

        # Telemetry data
        if msg_type == "telemetry":
            # These should be noted and ignored.
            print(f"Received telemetry sent at {sent_time}")

        elif msg_type == "simulation_instruction":
            # Parse the instruction, send to proxy for simulation tool
            # (for now, simply a sender), and send the results to Azure
            print(f"Received simulation instructions at {sent_time}")
            instr = data["instruction"]["pressure"]
            print(f"Instruction {instr}")
            sender.send_data(instr)

        elif msg_type == "simulation_result":
            # Take note that simulation results have been received
            print(f"Received simulation results sent at {sent_time}")
            vals = data["values"]["pressure"]
            print(f"Result {vals}")


# This function is called whenever an error happens
def on_error(particion_context: int, error: Exception):
    print(f"Error received: {error}")


def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    client = EventHubConsumerClient.from_connection_string(
        conn_str=config["access_receive"],
        consumer_group="$default",
    )

    # Listen perpetually to new messages
    try:
        with client:
            client.receive_batch(
                on_event_batch=on_events,
                on_error=on_error,
            )

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
