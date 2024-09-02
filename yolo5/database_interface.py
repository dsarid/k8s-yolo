import boto3
import polybot_helper_lib
try:
    import pymongo
except ImportError:
    print("pymongo is not installed")


def put_item(database_type, database_object, item, table_name, collection=None):
    if database_type == "DYNAMODB":
        return database_object.put_item(
            TableName=table_name,
            Item=polybot_helper_lib.dict_to_dynamo_format(item)
        )
    elif database_type == "MONGODB":
        myclient = database_object
        mydb = myclient[table_name]
        mycol = mydb[collection]

        x = mycol.insert_one(item)
        return x
