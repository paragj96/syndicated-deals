{
  "engines": {
    "composer": "^0.16.0"
  },
  "name": "syndicated-loan",
  "version": "0.2.0-20180102082548",
  "description": "EFFICIENT SYNDICATED DEAL CREDIT RISK MANAGEMENT USING MACHINE LEARNING ON BLOCKCHAIN",
  "networkImage": "https://hyperledger.github.io/composer-sample-networks/packages/basic-sample-network/networkimage.svg",
  "networkImageanimated": "https://hyperledger.github.io/composer-sample-networks/packages/basic-sample-network/networkimageanimated.svg",
  "scripts": {
    "prepublish": "mkdirp ./dist && composer archive create --sourceType dir --sourceName . -a ./dist/basic-sample-network.bna",
    "pretest": "npm run lint",
    "lint": "eslint .",
    "postlint": "npm run licchk",
    "licchk": "license-check",
    "postlicchk": "npm run doc",
    "doc": "jsdoc --pedantic --recurse -c jsdoc.json",
    "test-inner": "mocha -t 0 --recursive && cucumber-js",
    "test-cover": "nyc npm run test-inner",
    "test": "npm run test-inner"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/hyperledger/composer-sample-networks.git"
  },
  "keywords": [
    "sample",
    "composer",
    "composer-network"
  ],
  "author": "Hyperledger Composer",
  "license": "Apache-2.0",
  "devDependencies": {
    "chai": "^3.5.0",
    "chai-as-promised": "^6.0.0",
    "composer-admin": "^0.16.0",
    "composer-cli": "^0.16.0",
    "composer-client": "^0.16.0",
    "composer-connector-embedded": "^0.16.0",
    "composer-cucumber-steps": "^0.16.0",
    "cucumber": "^2.2.0",
    "eslint": "^3.6.1",
    "istanbul": "^0.4.5",
    "jsdoc": "^3.5.5",
    "license-check": "^1.1.5",
    "mkdirp": "^0.5.1",
    "mocha": "^3.2.0",
    "moment": "^2.17.1",
    "nyc": "^11.0.2"
  },
  "license-check-config": {
    "src": [
      "**/*.js",
      "!./coverage/**/*",
      "!./node_modules/**/*",
      "!./out/**/*",
      "!./scripts/**/*"
    ],
    "path": "header.txt",
    "blocking": true,
    "logInfo": false,
    "logError": true
  },
  "nyc": {
    "exclude": [
      "coverage/**",
      "features/**",
      "out/**",
      "test/**"
    ],
    "reporter": [
      "text-summary",
      "html"
    ],
    "all": true,
    "check-coverage": true,
    "statements": 100,
    "branches": 100,
    "functions": 100,
    "lines": 100
  }
}
