import json
import AWSIoTPythonSDK.MQTTLib as AWSIoTPyMQTT

# Define ENDPOINT, CLIENT_ID, PATH_TO_CERTIFICATE, PATH_TO_PRIVATE_KEY, PATH_TO_AMAZON_ROOT_CA_1, MESSAGE, TOPIC, and RANGE
ENDPOINT = "a3cmio73wih2yp-ats.iot.us-east-1.amazonaws.com"
CLIENT_ID = "Jetson-Nano"
PATH_TO_CERTIFICATE = "./keys/traffic-cloud.cert.pem"
PATH_TO_PRIVATE_KEY = "./keys/traffic-cloud.private.key"
PATH_TO_AMAZON_ROOT_CA_1 = "./keys/root-CA.crt"
#define the topic in AWS IoT policy
TOPIC = "trafficCloud/zone1" # Topic to which we are sending messages

# AWSIoTMQTTClient connection configuration
myAWSIoTMQTTClient = AWSIoTPyMQTT.AWSIoTMQTTClient(CLIENT_ID)

# AWSIoTMQTTClient connection configuration
myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
myAWSIoTMQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
myAWSIoTMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
myAWSIoTMQTTClient.configureConnectDisconnectTimeout(15)  # 15 sec
myAWSIoTMQTTClient.configureMQTTOperationTimeout(10)  # 10 sec


def connect_client():
	myAWSIoTMQTTClient.configureEndpoint(ENDPOINT, 8883)
	myAWSIoTMQTTClient.configureCredentials(PATH_TO_AMAZON_ROOT_CA_1, PATH_TO_PRIVATE_KEY, PATH_TO_CERTIFICATE)
	myAWSIoTMQTTClient.connect()

def publish_data(message):
	#topic, message and quality of service
	myAWSIoTMQTTClient.publish(TOPIC, json.dumps(message), 1) 
	print("Published: '" + json.dumps(message) + "' to the topic: " + TOPIC)

def disconnect_client():
	print('Publish End')
	myAWSIoTMQTTClient.disconnect()

