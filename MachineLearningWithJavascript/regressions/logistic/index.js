require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const LogisticRegression = require('./logistic-regression')
const plot = require('node-remote-plot')

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    converters: {
        passedemissions: (value) => value === 'TRUE' ? 1 : 0
    }
})

const regression = new LogisticRegression(features, labels, {
    learningRate: .5,
    iterations: 100,
    batchSize: 50,
    decisionBoundary: 0.6
})

regression.train()
// //regression.features.print()
console.log(regression.test(testFeatures, testLabels))

// plot({
//     x: regression.mseHistory.reverse(),
//     xLabel: 'Iteration #',
//     yLabel: 'Mean Squared Error'
// })

// console.log('r2=', r2)


// //console.log('Updated M is:', regression.weights.get(1, 0), 'Updated B is:', regression.weights.get(0, 0))