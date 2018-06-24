try:
    print('unzipping requirements')
    import unzip_requirements
except ImportError:
    print('in importerror export block')
    pass

print('starting imports')
import json
import sagemaker
import pandas


def hello(event, context):
    print('starting hello')
    
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    # return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }


if __name__ == "__main__":
    hello('', '')
