require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const LogisticRegression = require('./logistic-regression')
const plot = require('node-remote-plot')
const _ = require('lodash')

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg'],
    converters: {
        mpg: value => {
            const mpg = parseFloat(value)

            if (mpg < 15) {
                return [1, 0, 0]
            } else if (mpg < 30) {
                return [0, 1, 0]
            } else {
                return [0, 0, 1]
            }
        }
    }
})

const regression = new LogisticRegression(features, _.flatMap(labels), {
    learningRate: .5,
    iterations: 100,
    batchSize: 10
})

regression.train()

regression.predict([[150, 200, 2.223]]).print()

// // //regression.features.print()
// console.log(regression.test(testFeatures, testLabels))

// plot({
//     x: regression.costHistory.reverse(),

// })

// console.log('r2=', r2)


//console.log('Updated M is:', regression.weights.get(1, 0), 'Updated B is:', regression.weights.get(0, 0))