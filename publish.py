import json
import time
from awscrt import mqtt
from awsiot import mqtt_connection_builder


# Define ENDPOINT, CLIENT_ID, PATH_TO_CERTIFICATE, PATH_TO_PRIVATE_KEY, PATH_TO_AMAZON_ROOT_CA_1, MESSAGE, TOPIC, and RANGE
ENDPOINT = "a3cmio73wih2yp-ats.iot.us-east-1.amazonaws.com"
CLIENT_ID = "Jetson-Nano"
PATH_TO_CERTIFICATE = "./keys/traffic-cloud.cert.pem"
PATH_TO_PRIVATE_KEY = "./keys/traffic-cloud.private.key"
PATH_TO_AMAZON_ROOT_CA_1 = "./keys/root-CA.crt"
#define the topic in AWS IoT policy
TOPIC = "trafficCloud/zone1" # Topic to which we are sending messages


# MQTT connection
mqtt_connection = None

def connect_client():
    global mqtt_connection
    # Build the MQTT connection
    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=ENDPOINT,
        cert_filepath=PATH_TO_CERTIFICATE,
        pri_key_filepath=PATH_TO_PRIVATE_KEY,
        ca_filepath=PATH_TO_AMAZON_ROOT_CA_1,
        client_id=CLIENT_ID,
        clean_session=False,
        keep_alive_secs=30,
    )

    # Connect to AWS IoT Core
    print(f"Connecting to {ENDPOINT} with client ID '{CLIENT_ID}'...")
    connect_future = mqtt_connection.connect()
    connect_future.result()  # Wait for the connection to succeed
    print("Connected!")

def publish_data(message):
    # Publish the message to the topic
    message_json = json.dumps(message)
    mqtt_connection.publish(
        topic=TOPIC,
        payload=message_json,
        qos=mqtt.QoS.AT_LEAST_ONCE,
    )
    print(f"Published: {message_json} to the topic: {TOPIC}")

def disconnect_client():
    # Disconnect from AWS IoT Core
    print("Disconnecting...")
    disconnect_future = mqtt_connection.disconnect()
    disconnect_future.result()
    print("Disconnected!")
