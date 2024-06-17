class ComputeNode {
    constructor() {
        this.peerjsId = 'webgpt-' + Math.random().toString(36).substr(2, 8);
        this.peerjsNode = new Peer(this.peerjsId);
        this.peers = [];
        this.commandResolvers = {};
        this.peerjsNode.on('open', (id) => { 
            document.getElementById('peerjsId').innerText = id;
        })
        this.peerjsNode.on('connection', (conn) => {
            this.addConnection(conn);
        });
    }

    async onDataCommand(conn, dataCommand) {
        if (dataCommand._command === "runLayersForOthers") {
            console.log("TRYING TO RUN LAYER...")
            console.log(dataCommand);
            const {
                intermediateBufferSerialized,
                residualBufferSerialized,
            } = await window.model.runLayersForOthers({
                from: dataCommand.from,
                to: dataCommand.to,
                seq_length: dataCommand.seq_length,
                intermediateBufferSerialized: dataCommand.intermediateBufferSerialized,
                residualBufferSerialized: dataCommand.residualBufferSerialized,
            });
            console.log("FINISHED RUNNING FOR OTHERS...")
            conn.send({
                _command: "response",
                _id: dataCommand._id,
                intermediateBufferSerialized,
                residualBufferSerialized,
            });
        } else if (dataCommand._command === "response") {
            console.log("GOT RESPONSE FROM PEER ABOUT LAYER...")
            this.commandResolvers[dataCommand._id]({
                intermediateBufferSerialized: dataCommand.intermediateBufferSerialized,
                residualBufferSerialized: dataCommand.residualBufferSerialized,
            });
        }
    }

    updatePeersUi() {
        console.log("New connections:")
        console.log(this.peers);
        document.getElementById('peersList').innerHTML = this.peers.map(it => `- ${it.peer}`);
    }

    connectTo(targetId) {
        const conn = this.peerjsNode.connect(targetId);
        conn.on('open', () => {
            this.addConnection(conn);
        })
    }
    addConnection(conn) {
        this.peers.push(conn);
        conn.on('close', () => {
            this.peers = this.peers.filter(p => p !== conn);
            this.updatePeersUi();
        })
        conn.on('data', async (dataCommand) => {
            this.onDataCommand(conn, dataCommand);
        })
        this.updatePeersUi();
    }

    async runLayersOnAnyNode({from, to, seq_length, intermediateBufferSerialized, residualBufferSerialized}) {
        if (this.peers.length < 1) throw new Error("No peers connected");
        const cmdId = Math.random().toString(36).substr(2, 8);
        console.log("RUNNING LAYERS ON NODE...")
        this.peers[0].send({
            _command: "runLayersForOthers",
            _id: cmdId,
            from,
            to,
            seq_length,
            intermediateBufferSerialized,
            residualBufferSerialized,
        })
        return new Promise((resolve, reject) => {
            this.commandResolvers[cmdId] = resolve;
        })
    }
}

window.computeNode = new ComputeNode();