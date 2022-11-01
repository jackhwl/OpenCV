require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear-regression')
const plot = require('node-remote-plot')

let { features, labels, testFeatures, testLabels } = loadCSV('./data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
})

const regression = new LinearRegression(features, labels, {
    learningRate: .1,
    iterations: 4,
    batchSize: 1
})

regression.train()
//regression.features.print()
const r2 = regression.test(testFeatures, testLabels)

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error'
})

console.log('r2=', r2)
//console.log('Updated M is:', regression.weights.get(1, 0), 'Updated B is:', regression.weights.get(0, 0))