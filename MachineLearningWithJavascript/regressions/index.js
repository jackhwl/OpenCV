require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear-regression')

let { features, labels, testFeatures, testLabels } = loadCSV('./data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
})

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 100
})

regression.train()
//regression.features.print()
const r2 = regression.test(testFeatures, testLabels)
console.log('r2=', r2)
//console.log('Updated M is:', regression.weights.get(1, 0), 'Updated B is:', regression.weights.get(0, 0))