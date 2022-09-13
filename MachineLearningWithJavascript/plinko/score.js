const outputs = []

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
    outputs.push([dropPosition, bounciness, size, bucketLabel])
}

function runAnalysis() {
    const testSetSize = 10
    const [testSet, trainingSet] = splitDataset(outputs, testSetSize)
    // let numberCorrect = 0
    // for (let i=0; i < testSet.length; i++) {
    //     const bucket = knn(trainingSet, testSet[i][0])
    //     if (bucket === testSet[i][3]) {
    //         numberCorrect++
    //     }
    // }
    _.range(1, 15).forEach(k => {
        const accuracy = _.chain(testSet)
            .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === testPoint[3])
            .size()
            .divide(testSetSize)
            .value()

        console.log('For k of ', k, 'accuracy is :', accuracy)
    })
}

function knn(data, point, k) {
    return _.chain(data)
        .map(row => [distance(_.initial(row), point), _.last(row)])
        .sortBy(row => row[0])
        .slice(0, k)
        .countBy(row => row[1])
        .toPairs()
        .sortBy(row => row[1])
        .last()
        .first()
        .parseInt()
        .value()
}

function distance(pointA, pointB) {
    return _.chain(pointA)
        .zip(pointB)
        .map(([a, b]) => (a - b) ** 2)
        .sum()
        .value() ** 0.5
}

function splitDataset(data, testCount) {
    const shuffled = _.shuffle(data)
    
    const testSet = _.slice(shuffled, 0, testCount)
    const trainingSet = _.slice(shuffled, testCount)
    
    return [testSet, trainingSet]
}

function minMax(data, featureCount) {
    const clonedData = _.cloneDeep(data)

    for (let i = 0; i < featureCount; i++) {
        const column = clonedData.map(row => row[i])

        const min = _.min(column)
        const max = _.max(column)

        for (let j = 0; j < clonedData.length; j++) {
            clonedData[j][i] = (clonedData[j][i] - min) / (max - min)
        }
    }

    return clonedData
}