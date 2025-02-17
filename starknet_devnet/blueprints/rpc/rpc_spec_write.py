# pylint: disable=too-many-lines, missing-module-docstring
RPC_SPECIFICATION_WRITE = r"""
{
    "openrpc": "1.0.0-rc1",
    "info": {
        "version": "0.4.0",
        "title": "Starknet Node Write API",
        "license": {}
    },
    "servers": [],
    "methods": [
        {
            "name": "starknet_addInvokeTransaction",
            "summary": "Submit a new transaction to be added to the chain",
            "params": [
                {
                    "name": "invoke_transaction",
                    "description": "The information needed to invoke the function (or account, for version 1 transactions)",
                    "required": true,
                    "schema": {
                        "$ref": "#/components/schemas/BROADCASTED_INVOKE_TXN"
                    }
                }
            ],
            "result": {
                "name": "result",
                "description": "The result of the transaction submission",
                "schema": {
                    "type": "object",
                    "properties": {
                        "transaction_hash": {
                            "title": "The hash of the invoke transaction",
                            "$ref": "#/components/schemas/TXN_HASH"
                        }
                    }
                },
                "required": ["transaction_hash"]
            },
            "errors": []
        },
        {
            "name": "starknet_addDeclareTransaction",
            "summary": "Submit a new class declaration transaction",
            "params": [
                {
                    "name": "declare_transaction",
                    "schema": {
                        "$ref": "#/components/schemas/BROADCASTED_DECLARE_TXN"
                    }
                }
            ],
            "result": {
                "name": "result",
                "description": "The result of the transaction submission",
                "schema": {
                    "type": "object",
                    "properties": {
                        "transaction_hash": {
                            "title": "The hash of the declare transaction",
                            "$ref": "#/components/schemas/TXN_HASH"
                        },
                        "class_hash": {
                            "title": "The hash of the declared class",
                            "$ref": "#/components/schemas/FELT"
                        }
                    }
                },
                "required": ["transaction_hash", "class_hash"]
            },
            "errors": [
                {
                    "$ref": "#/components/errors/INVALID_CONTRACT_CLASS"
                }
            ]
        },
        {
            "name": "starknet_addDeployTransaction",
            "summary": "Submit a new deploy contract transaction",
            "params": [
                {
                    "name": "deploy_transaction",
                    "description": "The deploy transaction",
                    "schema": {
                        "$ref": "#/components/schemas/BROADCASTED_DEPLOY_TXN"
                    }
                }
            ],
            "result": {
                "name": "result",
                "description": "The result of the transaction submission",
                "schema": {
                    "type": "object",
                    "properties": {
                        "transaction_hash": {
                            "title": "The hash of the deploy transaction",
                            "$ref": "#/components/schemas/TXN_HASH"
                        },
                        "contract_address": {
                            "title": "The address of the new contract",
                            "$ref": "#/components/schemas/FELT"
                        }
                    },
                    "required": ["transaction_hash", "contract_address"]
                }
            },
            "errors": [
                {
                    "$ref": "#/components/errors/INVALID_CONTRACT_CLASS"
                }
            ]
        },
        {
            "name": "starknet_addDeployAccountTransaction",
            "summary": "Submit a new deploy account transaction",
            "params": [
                {
                    "name": "deploy_account_transaction",
                    "description": "The deploy account transaction",
                    "schema": {
                        "$ref": "#/components/schemas/BROADCASTED_DEPLOY_ACCOUNT_TXN"
                    }
                }
            ],
            "result": {
                "name": "result",
                "description": "The result of the transaction submission",
                "schema": {
                    "type": "object",
                    "properties": {
                        "transaction_hash": {
                            "title": "The hash of the deploy transaction",
                            "$ref": "#/components/schemas/TXN_HASH"
                        },
                        "contract_address": {
                            "title": "The address of the new contract",
                            "$ref": "#/components/schemas/FELT"
                        }
                    },
                    "required": ["transaction_hash", "contract_address"]
                }
            },
            "errors": [
                {
                    "$ref": "#/components/errors/CLASS_HASH_NOT_FOUND"
                }
            ]
        }
    ],
    "components": {
        "contentDescriptors": {},
        "schemas": {
            "CONTRACT_CLASS": {
                "$ref": "./api/starknet_api_openrpc.json#/components/schemas/CONTRACT_CLASS"
            },
            "NUM_AS_HEX": {
                "title": "An integer number in hex format (0x...)",
                "type": "string",
                "pattern": "^0x[a-fA-F0-9]+$"
            },
            "SIGNATURE": {
                "$ref": "./api/starknet_api_openrpc.json#/components/schemas/SIGNATURE"
            },
            "FELT": {
                "$ref": "./api/starknet_api_openrpc.json#/components/schemas/FELT"
            },
            "TXN_HASH": {
                "$ref": "./api/starknet_api_openrpc.json#/components/schemas/TXN_HASH"
            },
            "BROADCASTED_INVOKE_TXN": {
                "$ref": "./api/starknet_api_openrpc.json#/components/schemas/BROADCASTED_INVOKE_TXN"
            },
            "BROADCASTED_DECLARE_TXN": {
                "$ref": "./api/starknet_api_openrpc.json#/components/schemas/BROADCASTED_DECLARE_TXN"
            },
            "BROADCASTED_DEPLOY_TXN": {
                "$ref": "./api/starknet_api_openrpc.json#/components/schemas/BROADCASTED_DEPLOY_TXN"
            },
            "BROADCASTED_DEPLOY_ACCOUNT_TXN": {
                "$ref": "./api/starknet_api_openrpc.json#/components/schemas/BROADCASTED_DEPLOY_ACCOUNT_TXN"
            },
            "FUNCTION_CALL": {
                "$ref": "./api/starknet_api_openrpc.json#/components/schemas/FUNCTION_CALL"
            }
        },
        "errors": {
            "INVALID_CONTRACT_CLASS": {
                "code": 50,
                "message": "Invalid contract class"
            },
            "CLASS_HASH_NOT_FOUND": {
                "$ref": "./api/starknet_api_openrpc.json#/components/errors/CLASS_HASH_NOT_FOUND"
            }
        }
    }
}
"""
