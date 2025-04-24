import os
import json
import socket
import traceback
from qgis.core import *
from qgis.gui import *
from qgis.PyQt.QtCore import QObject, pyqtSignal, QTimer, Qt, QSize
from qgis.PyQt.QtWidgets import QAction, QDockWidget, QVBoxLayout, QLabel, QPushButton, QSpinBox, QWidget
from qgis.PyQt.QtGui import QIcon, QColor
from qgis.utils import active_plugins

class QgisMCPServer(QObject):
    """Server class to handle socket connections and execute QGIS commands"""
    
    def __init__(self, host='localhost', port=9876, iface=None):
        super().__init__()
        self.host = host
        self.port = port
        self.iface = iface
        self.running = False
        self.socket = None
        self.client = None
        self.buffer = b''
        self.timer = None
    
    def start(self):
        """Start the server"""
        self.running = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.setblocking(False)
            
            # Create a timer to process server operations
            self.timer = QTimer()
            self.timer.timeout.connect(self.process_server)
            self.timer.start(100)  # 100ms interval
            
            QgsMessageLog.logMessage(f"QGIS MCP server started on {self.host}:{self.port}", "QGIS MCP")
            return True
        except Exception as e:
            QgsMessageLog.logMessage(f"Failed to start server: {str(e)}", "QGIS MCP", Qgis.Critical)
            self.stop()
            return False
            
    def stop(self):
        """Stop the server"""
        self.running = False
        
        if self.timer:
            self.timer.stop()
            self.timer = None
            
        if self.socket:
            self.socket.close()
        if self.client:
            self.client.close()
            
        self.socket = None
        self.client = None
        QgsMessageLog.logMessage("QGIS MCP server stopped", "QGIS MCP")
    
    def process_server(self):
        """Process server operations (called by timer)"""
        if not self.running:
            return
            
        try:
            # Accept new connections
            if not self.client and self.socket:
                try:
                    self.client, address = self.socket.accept()
                    self.client.setblocking(False)
                    QgsMessageLog.logMessage(f"Connected to client: {address}", "QGIS MCP")
                except BlockingIOError:
                    pass  # No connection waiting
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error accepting connection: {str(e)}", "QGIS MCP", Qgis.Warning)
                
            # Process existing connection
            if self.client:
                try:
                    # Try to receive data
                    try:
                        data = self.client.recv(8192)
                        if data:
                            self.buffer += data
                            # Try to process complete messages
                            try:
                                # Attempt to parse the buffer as JSON
                                command = json.loads(self.buffer.decode('utf-8'))
                                # If successful, clear the buffer and process command
                                self.buffer = b''
                                response = self.execute_command(command)
                                response_json = json.dumps(response)
                                self.client.sendall(response_json.encode('utf-8'))
                            except json.JSONDecodeError:
                                # Incomplete data, keep in buffer
                                pass
                        else:
                            # Connection closed by client
                            QgsMessageLog.logMessage("Client disconnected", "QGIS MCP")
                            self.client.close()
                            self.client = None
                            self.buffer = b''
                    except BlockingIOError:
                        pass  # No data available
                    except Exception as e:
                        QgsMessageLog.logMessage(f"Error receiving data: {str(e)}", "QGIS MCP", Qgis.Warning)
                        self.client.close()
                        self.client = None
                        self.buffer = b''
                        
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error with client: {str(e)}", "QGIS MCP", Qgis.Warning)
                    if self.client:
                        self.client.close()
                        self.client = None
                    self.buffer = b''
                    
        except Exception as e:
            QgsMessageLog.logMessage(f"Server error: {str(e)}", "QGIS MCP", Qgis.Critical)

    def execute_command(self, command):
        """Execute a command"""
        try:
            cmd_type = command.get("type")
            params = command.get("params", {})
            
            handlers = {
                "ping": self.ping,
                "get_qgis_info": self.get_qgis_info,
                "load_project": self.load_project,
                "get_project_info": self.get_project_info,
                "execute_code": self.execute_code,
                "add_vector_layer": self.add_vector_layer,
                "add_raster_layer": self.add_raster_layer,
                "get_layers": self.get_layers,
                "remove_layer": self.remove_layer,
                "zoom_to_layer": self.zoom_to_layer,
                "get_layer_features": self.get_layer_features,
                "execute_processing": self.execute_processing,
                "save_project": self.save_project,
                "render_map": self.render_map,
                "create_new_project": self.create_new_project,
            }
            
            handler = handlers.get(cmd_type)
            if handler:
                try:
                    QgsMessageLog.logMessage(f"Executing handler for {cmd_type}", "QGIS MCP")
                    result = handler(**params)
                    QgsMessageLog.logMessage(f"Handler execution complete", "QGIS MCP")
                    return {"status": "success", "result": result}
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error in handler: {str(e)}", "QGIS MCP", Qgis.Critical)
                    traceback.print_exc()
                    return {"status": "error", "message": str(e)}
            else:
                return {"status": "error", "message": f"Unknown command type: {cmd_type}"}
                
        except Exception as e:
            QgsMessageLog.logMessage(f"Error executing command: {str(e)}", "QGIS MCP", Qgis.Critical)
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    # Command handlers
    def ping(self, **kwargs):
        """Simple ping command"""
        return {"pong": True}
    
    def get_qgis_info(self, **kwargs):
        """Get basic QGIS information"""
        return {
            "qgis_version": Qgis.version(),
            "profile_folder": QgsApplication.qgisSettingsDirPath(),
            "plugins_count": len(active_plugins)
        }
    
    def get_project_info(self, **kwargs):
        """Get information about the current QGIS project"""
        project = QgsProject.instance()
        
        # Get basic project information
        info = {
            "filename": project.fileName(),
            "title": project.title(),
            "layer_count": len(project.mapLayers()),
            "crs": project.crs().authid(),
            "layers": []
        }
        
        # Add basic layer information (limit to 10 layers for performance)
        layers = list(project.mapLayers().values())
        for i, layer in enumerate(layers):
            if i >= 10:  # Limit to 10 layers
                break
                
            layer_info = {
                "id": layer.id(),
                "name": layer.name(),
                "type": self._get_layer_type(layer),
                "visible": layer.isValid() and project.layerTreeRoot().findLayer(layer.id()).isVisible()
            }
            info["layers"].append(layer_info)
        
        return info
    
    def _get_layer_type(self, layer):
        """Helper to get layer type as string"""
        if layer.type() == QgsMapLayer.VectorLayer:
            return f"vector_{layer.geometryType()}"
        elif layer.type() == QgsMapLayer.RasterLayer:
            return "raster"
        else:
            return str(layer.type())
    
    def execute_code(self, code, **kwargs):
        """Execute arbitrary PyQGIS code"""
        try:
            # Create a local namespace for execution
            namespace = {
                "qgis": Qgis,
                "QgsProject": QgsProject,
                "iface": self.iface,
                "QgsApplication": QgsApplication,
                "QgsVectorLayer": QgsVectorLayer,
                "QgsRasterLayer": QgsRasterLayer,
                "QgsCoordinateReferenceSystem": QgsCoordinateReferenceSystem
            }
            
            # Execute the code
            exec(code, namespace)
            return {"executed": True}
        except Exception as e:
            raise Exception(f"Code execution error: {str(e)}")
    
    def add_vector_layer(self, path, name=None, provider="ogr", **kwargs):
        """Add a vector layer to the project"""
        if not name:
            name = os.path.basename(path)
            
        # Create the layer
        layer = QgsVectorLayer(path, name, provider)
        
        if not layer.isValid():
            raise Exception(f"Layer is not valid: {path}")
        
        # Add to project
        QgsProject.instance().addMapLayer(layer)
        
        return {
            "id": layer.id(),
            "name": layer.name(),
            "type": self._get_layer_type(layer),
            "feature_count": layer.featureCount()
        }
    
    def add_raster_layer(self, path, name=None, provider="gdal", **kwargs):
        """Add a raster layer to the project"""
        if not name:
            name = os.path.basename(path)
            
        # Create the layer
        layer = QgsRasterLayer(path, name, provider)
        
        if not layer.isValid():
            raise Exception(f"Layer is not valid: {path}")
        
        # Add to project
        QgsProject.instance().addMapLayer(layer)
        
        return {
            "id": layer.id(),
            "name": layer.name(),
            "type": "raster",
            "width": layer.width(),
            "height": layer.height()
        }
    
    def get_layers(self, **kwargs):
        """Get all layers in the project"""
        project = QgsProject.instance()
        layers = []
        
        for layer_id, layer in project.mapLayers().items():
            layer_info = {
                "id": layer_id,
                "name": layer.name(),
                "type": self._get_layer_type(layer),
                "visible": project.layerTreeRoot().findLayer(layer_id).isVisible()
            }
            
            # Add type-specific information
            if layer.type() == QgsMapLayer.VectorLayer:
                layer_info.update({
                    "feature_count": layer.featureCount(),
                    "geometry_type": layer.geometryType()
                })
            elif layer.type() == QgsMapLayer.RasterLayer:
                layer_info.update({
                    "width": layer.width(),
                    "height": layer.height()
                })
                
            layers.append(layer_info)
        
        return layers
    
    def remove_layer(self, layer_id, **kwargs):
        """Remove a layer from the project"""
        project = QgsProject.instance()
        
        if layer_id in project.mapLayers():
            project.removeMapLayer(layer_id)
            return {"removed": layer_id}
        else:
            raise Exception(f"Layer not found: {layer_id}")
    
    def zoom_to_layer(self, layer_id, **kwargs):
        """Zoom to a layer's extent"""
        project = QgsProject.instance()
        
        if layer_id in project.mapLayers():
            layer = project.mapLayer(layer_id)
            self.iface.setActiveLayer(layer)
            self.iface.zoomToActiveLayer()
            return {"zoomed_to": layer_id}
        else:
            raise Exception(f"Layer not found: {layer_id}")
    
    def get_layer_features(self, layer_id, limit=10, **kwargs):
        """Get features from a vector layer"""
        project = QgsProject.instance()
        
        if layer_id in project.mapLayers():
            layer = project.mapLayer(layer_id)
            
            if layer.type() != QgsMapLayer.VectorLayer:
                raise Exception(f"Layer is not a vector layer: {layer_id}")
            
            features = []
            for i, feature in enumerate(layer.getFeatures()):
                if i >= limit:
                    break
                    
                # Extract attributes
                attrs = {}
                for field in layer.fields():
                    attrs[field.name()] = feature.attribute(field.name())
                
                # Extract geometry if available
                geom = None
                if feature.hasGeometry():
                    geom = {
                        "type": feature.geometry().type(),
                        "wkt": feature.geometry().asWkt(precision=4)
                    }
                
                features.append({
                    "id": feature.id(),
                    "attributes": attrs,
                    "geometry": geom
                })
            
            return {
                "layer_id": layer_id,
                "feature_count": layer.featureCount(),
                "features": features,
                "fields": [field.name() for field in layer.fields()]
            }
        else:
            raise Exception(f"Layer not found: {layer_id}")
    
    def execute_processing(self, algorithm, parameters, **kwargs):
        """Execute a processing algorithm"""
        try:
            import processing
            result = processing.run(algorithm, parameters)
            return {
                "algorithm": algorithm,
                "result": {k: str(v) for k, v in result.items()}  # Convert values to strings for JSON
            }
        except Exception as e:
            raise Exception(f"Processing error: {str(e)}")
    
    def save_project(self, path=None, **kwargs):
        """Save the current project"""
        project = QgsProject.instance()
        
        if not path and not project.fileName():
            raise Exception("No project path specified and no current project path")
        
        save_path = path if path else project.fileName()
        if project.write(save_path):
            return {"saved": save_path}
        else:
            raise Exception(f"Failed to save project to {save_path}")
    
    def load_project(self, path, **kwargs):
        """Load a project"""
        project = QgsProject.instance()
        
        if project.read(path):
            self.iface.mapCanvas().refresh()
            return {
                "loaded": path,
                "layer_count": len(project.mapLayers())
            }
        else:
            raise Exception(f"Failed to load project from {path}")
    
    def create_new_project(self, path, **kwargs):
        """
        Creates a new QGIS project and saves it at the specified path.
        If a project is already loaded, it clears it before creating the new one.
        
        :param project_path: Full path where the project will be saved
                            (e.g., 'C:/path/to/project.qgz')
        """
        project = QgsProject.instance()
        
        if project.fileName():
            project.clear()
        
        project.setFileName(path)
        self.iface.mapCanvas().refresh()
        
        # Save the project
        if project.write():
            return {
                "created": f"Project created and saved successfully at: {path}",
                "layer_count": len(project.mapLayers())
            }
        else:
            raise Exception(f"Failed to save project to {path}")
    
    def render_map(self, path, width=800, height=600, **kwargs):
        """Render the current map view to an image"""
        try:
            # Create map settings
            ms = QgsMapSettings()
            
            # Set layers to render
            layers = list(QgsProject.instance().mapLayers().values())
            ms.setLayers(layers)
            
            # Set map canvas properties
            rect = self.iface.mapCanvas().extent()
            ms.setExtent(rect)
            ms.setOutputSize(QSize(width, height))
            ms.setBackgroundColor(QColor(255, 255, 255))
            ms.setOutputDpi(96)
            
            # Create the render
            render = QgsMapRendererParallelJob(ms)
            
            # Start rendering
            render.start()
            render.waitForFinished()
            
            # Get the image and save
            img = render.renderedImage()
            if img.save(path):
                return {
                    "rendered": True,
                    "path": path,
                    "width": width,
                    "height": height
                }
            else:
                raise Exception(f"Failed to save rendered image to {path}")
                
        except Exception as e:
            raise Exception(f"Render error: {str(e)}")

    def get_layer_schema(self, layer_id, **kwargs):
    """Get the schema of a layer"""
    project = QgsProject.instance()
    
    if layer_id in project.mapLayers():
        layer = project.mapLayer(layer_id)
        
        if layer.type() != QgsMapLayer.VectorLayer:
            raise Exception(f"Layer is not a vector layer: {layer_id}")
        
        # Get field information
        fields_info = []
        for field in layer.fields():
            field_info = {
                "name": field.name(),
                "type": field.typeName(),
                "length": field.length(),
                "precision": field.precision(),
                "comment": field.comment()
            }
            
            # Add constraints
            constraints = field.constraints()
            field_info["constraints"] = {
                "not_null": constraints.constraints() & QgsFieldConstraints.ConstraintNotNull,
                "unique": constraints.constraints() & QgsFieldConstraints.ConstraintUnique,
                "expression": constraints.constraintExpression()
            }
            
            fields_info.append(field_info)
        
        # Get primary key information
        primary_key_fields = layer.primaryKeyAttributes()
        pk_names = [layer.fields().at(idx).name() for idx in primary_key_fields]
        
        # Create schema information
        schema_info = {
            "layer_id": layer_id,
            "name": layer.name(),
            "fields": fields_info,
            "primary_key_fields": pk_names,
            "feature_count": layer.featureCount(),
            "geometry_type": layer.geometryType(),
            "wkb_type": layer.wkbType()
        }
        
        return schema_info
    else:
        raise Exception(f"Layer not found: {layer_id}")
        
def get_layer_features_extended(self, layer_id, limit=10, offset=0, filter_expression=None, 
                               order_by=None, fields=None, **kwargs):
    """Get features from a layer with advanced options"""
    project = QgsProject.instance()
    
    if layer_id in project.mapLayers():
        layer = project.mapLayer(layer_id)
        
        if layer.type() != QgsMapLayer.VectorLayer:
            raise Exception(f"Layer is not a vector layer: {layer_id}")
        
        # Build request
        request = QgsFeatureRequest()
        
        # Apply limit and offset
        request.setLimit(limit)
        if offset > 0:
            request.setOffset(offset)
        
        # Apply field restriction if specified
        if fields:
            field_indices = []
            for field_name in fields:
                idx = layer.fields().indexFromName(field_name)
                if idx >= 0:
                    field_indices.append(idx)
            
            if field_indices:
                request.setSubsetOfAttributes(field_indices)
        
        # Apply filter expression
        if filter_expression:
            expr = QgsExpression(filter_expression)
            if expr.hasParserError():
                raise Exception(f"Invalid filter expression: {expr.parserErrorString()}")
            request.setFilterExpression(filter_expression)
        
        # Apply ordering
        if order_by:
            order_by_clause = []
            for field in order_by:
                is_desc = field.startswith('-')
                field_name = field[1:] if is_desc else field
                idx = layer.fields().indexFromName(field_name)
                if idx >= 0:
                    order_by_clause.append(QgsFeatureRequest.OrderByClause(
                        field_name, 
                        ascending=not is_desc
                    ))
            
            if order_by_clause:
                order_by = QgsFeatureRequest.OrderBy(order_by_clause)
                request.setOrderBy(order_by)
        
        # Get total count matching filter
        total_count = 0
        if filter_expression:
            expr = QgsExpression(filter_expression)
            context = QgsExpressionContext()
            context.appendScope(QgsExpressionContextUtils.layerScope(layer))
            matching_count = 0
            for feature in layer.getFeatures():
                context.setFeature(feature)
                if expr.evaluate(context):
                    matching_count += 1
            total_count = matching_count
        else:
            total_count = layer.featureCount()
        
        # Fetch features
        features = []
        for feature in layer.getFeatures(request):
            # Extract attributes
            attrs = {}
            for field in layer.fields():
                if not fields or field.name() in fields:
                    attrs[field.name()] = feature.attribute(field.name())
            
            # Extract geometry if available
            geom = None
            if feature.hasGeometry():
                geom = {
                    "type": feature.geometry().type(),
                    "wkt": feature.geometry().asWkt(precision=4)
                }
            
            features.append({
                "id": feature.id(),
                "attributes": attrs,
                "geometry": geom
            })
        
        return {
            "layer_id": layer_id,
            "total_count": total_count,
            "returned_count": len(features),
            "offset": offset,
            "limit": limit,
            "features": features,
            "fields": [field.name() for field in layer.fields() if not fields or field.name() in fields]
        }
    else:
        raise Exception(f"Layer not found: {layer_id}")
        
def get_field_statistics(self, layer_id, field_name, filter_expression=None, **kwargs):
    """Get statistics for a field"""
    project = QgsProject.instance()
    
    if layer_id in project.mapLayers():
        layer = project.mapLayer(layer_id)
        
        if layer.type() != QgsMapLayer.VectorLayer:
            raise Exception(f"Layer is not a vector layer: {layer_id}")
        
        # Check if field exists
        field_idx = layer.fields().indexFromName(field_name)
        if field_idx < 0:
            raise Exception(f"Field not found: {field_name}")
        
        field = layer.fields().at(field_idx)
        field_type = field.type()
        
        # Build request
        request = QgsFeatureRequest()
        request.setSubsetOfAttributes([field_idx])
        
        # Apply filter expression
        if filter_expression:
            expr = QgsExpression(filter_expression)
            if expr.hasParserError():
                raise Exception(f"Invalid filter expression: {expr.parserErrorString()}")
            request.setFilterExpression(filter_expression)
        
        # Collect values
        values = []
        for feature in layer.getFeatures(request):
            value = feature.attribute(field_name)
            if value:
                values.append(value)
        
        stats = {
            "layer_id": layer_id,
            "field_name": field_name,
            "count": len(values),
            "unique_count": len(set(values))
        }
        
        # Calculate statistics based on field type
        if field_type in [QVariant.Int, QVariant.LongLong, QVariant.Double]:
            # Numeric field
            if values:
                numeric_values = [float(v) for v in values if v is not None]
                if numeric_values:
                    stats.update({
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "sum": sum(numeric_values),
                        "mean": sum(numeric_values) / len(numeric_values)
                    })
                    
                    # Calculate median
                    sorted_values = sorted(numeric_values)
                    n = len(sorted_values)
                    if n % 2 == 0:
                        median = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
                    else:
                        median = sorted_values[n//2]
                    stats["median"] = median
                    
                    # Calculate standard deviation
                    if len(numeric_values) > 1:
                        mean = stats["mean"]
                        variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
                        stats["std_dev"] = variance ** 0.5
        
        # Calculate frequency distribution for any field type
        if values:
            frequency = {}
            for value in values:
                value_str = str(value)
                if value_str in frequency:
                    frequency[value_str] += 1
                else:
                    frequency[value_str] = 1
            
            # Sort by frequency, descending
            frequency = {k: v for k, v in sorted(frequency.items(), key=lambda item: item[1], reverse=True)}
            
            # Limit to top 50 values for efficiency
            top_values = list(frequency.items())[:50]
            stats["frequency"] = {k: v for k, v in top_values}
        
        return stats
    else:
        raise Exception(f"Layer not found: {layer_id}")
        
def update_feature_attribute(self, layer_id, feature_id, field_name, value, **kwargs):
    """Update an attribute value for a specific feature"""
    project = QgsProject.instance()
    
    if layer_id in project.mapLayers():
        layer = project.mapLayer(layer_id)
        
        if layer.type() != QgsMapLayer.VectorLayer:
            raise Exception(f"Layer is not a vector layer: {layer_id}")
        
        # Check if field exists
        field_idx = layer.fields().indexFromName(field_name)
        if field_idx < 0:
            raise Exception(f"Field not found: {field_name}")
        
        # Check if the layer is editable
        if not layer.isEditable():
            layer.startEditing()
        
        # Update the attribute
        if layer.changeAttributeValue(feature_id, field_idx, value):
            # Commit the changes
            if layer.commitChanges():
                return {
                    "success": True,
                    "feature_id": feature_id,
                    "field_name": field_name,
                    "value": value
                }
            else:
                # Failed to commit, get the errors
                errors = layer.commitErrors()
                layer.rollBack()
                raise Exception(f"Failed to commit changes: {'. '.join(errors)}")
        else:
            layer.rollBack()
            raise Exception(f"Failed to update attribute value for feature {feature_id}")
    else:
        raise Exception(f"Layer not found: {layer_id}")
        
def create_thematic_map(self, layer_id, field_name, classification_method='equal_interval', 
                       num_classes=5, color_ramp='viridis', output_path=None, **kwargs):
    """Create a thematic map based on attribute values"""
    project = QgsProject.instance()
    
    if layer_id in project.mapLayers():
        layer = project.mapLayer(layer_id)
        
        if layer.type() != QgsMapLayer.VectorLayer:
            raise Exception(f"Layer is not a vector layer: {layer_id}")
        
        # Check if field exists
        field_idx = layer.fields().indexFromName(field_name)
        if field_idx < 0:
            raise Exception(f"Field not found: {field_name}")
        
        # Create renderer
        field = layer.fields().at(field_idx)
        
        # Map classification method to QGIS enum
        method_map = {
            'equal_interval': QgsClassificationEqualInterval,
            'quantile': QgsClassificationQuantile,
            'natural_breaks': QgsClassificationJenks,
            'standard_deviation': QgsClassificationStandardDeviation,
            'pretty_breaks': QgsClassificationPrettyBreaks
        }
        
        qgis_method = method_map.get(classification_method)
        if not qgis_method:
            raise Exception(f"Invalid classification method: {classification_method}. "
                           f"Valid options are: {', '.join(method_map.keys())}")
        
        # Create a color ramp
        style = QgsStyle.defaultStyle()
        ramp_names = style.colorRampNames()
        
        if color_ramp not in ramp_names:
            raise Exception(f"Invalid color ramp: {color_ramp}. "
                           f"Valid options include: {', '.join(ramp_names[:10])}...")
        
        color_ramp_obj = style.colorRamp(color_ramp)
        
        # Create a graduated renderer
        renderer = QgsGraduatedSymbolRenderer(field_name, [])
        renderer.setClassificationMethod(qgis_method())
        renderer.setClassAttribute(field_name)
        renderer.setGraduatedMethod(QgsGraduatedSymbolRenderer.GraduatedColor)
        renderer.setSourceColorRamp(color_ramp_obj)
        
        # Update the renderer with the classes
        renderer.updateClasses(layer, num_classes)
        
        # Apply the renderer to the layer
        layer.setRenderer(renderer)
        
        # Refresh the layer
        layer.triggerRepaint()
        
        # Render to image if requested
        result = {
            "success": True,
            "layer_id": layer_id,
            "field_name": field_name,
            "classification_method": classification_method,
            "num_classes": num_classes,
            "color_ramp": color_ramp,
            "classes": []
        }
        
        # Add class information
        for idx, range in enumerate(renderer.ranges()):
            result["classes"].append({
                "label": range.label(),
                "lower_value": range.lowerValue(),
                "upper_value": range.upperValue(),
                "color": range.symbol().color().name(),
                "symbol": range.symbol().symbolLayerCount()
            })
        
        # Render to image if requested
        if output_path:
            settings = QgsMapSettings()
            settings.setLayers([layer])
            settings.setExtent(layer.extent())
            settings.setOutputSize(QSize(800, 600))
            
            render = QgsMapRendererParallelJob(settings)
            render.start()
            render.waitForFinished()
            
            img = render.renderedImage()
            if img.save(output_path):
                result["output_path"] = output_path
            else:
                result["error"] = f"Failed to save rendered image to {output_path}"
        
        return result
    else:
        raise Exception(f"Layer not found: {layer_id}")
        
def update_features_by_expression(self, layer_id, field_name, expression, filter_expression=None, **kwargs):
    """Update attribute values for multiple features based on an expression"""
    project = QgsProject.instance()
    
    if layer_id in project.mapLayers():
        layer = project.mapLayer(layer_id)
        
        if layer.type() != QgsMapLayer.VectorLayer:
            raise Exception(f"Layer is not a vector layer: {layer_id}")
        
        # Check if field exists
        field_idx = layer.fields().indexFromName(field_name)
        if field_idx < 0:
            raise Exception(f"Field not found: {field_name}")
        
        # Check expression validity
        update_expr = QgsExpression(expression)
        if update_expr.hasParserError():
            raise Exception(f"Invalid update expression: {update_expr.parserErrorString()}")
        
        # Build request for features to update
        request = QgsFeatureRequest()
        
        # Apply filter expression if provided
        if filter_expression:
            filter_expr = QgsExpression(filter_expression)
            if filter_expr.hasParserError():
                raise Exception(f"Invalid filter expression: {filter_expr.parserErrorString()}")
            request.setFilterExpression(filter_expression)
        
        # Start editing
        if not layer.isEditable():
            layer.startEditing()
        
        # Create expression context
        context = QgsExpressionContext()
        context.appendScope(QgsExpressionContextUtils.layerScope(layer))
        
        # Update features
        updated_count = 0
        features = list(layer.getFeatures(request))
        total_features = len(features)
        
        for feature in features:
            context.setFeature(feature)
            new_value = update_expr.evaluate(context)
            
            if update_expr.hasEvalError():
                continue  # Skip features with evaluation errors
                
            if layer.changeAttributeValue(feature.id(), field_idx, new_value):
                updated_count += 1
        
        # Commit changes
        if layer.commitChanges():
            return {
                "success": True,
                "layer_id": layer_id,
                "field_name": field_name,
                "total_features": total_features,
                "updated_features": updated_count
            }
        else:
            # Failed to commit, get the errors
            errors = layer.commitErrors()
            layer.rollBack()
            raise Exception(f"Failed to commit changes: {'. '.join(errors)}")
    else:
        raise Exception(f"Layer not found: {layer_id}")
        
def export_attribute_data(self, layer_id, output_path, format='csv', fields=None, filter_expression=None, **kwargs):
    """Export attribute data to a file"""
    project = QgsProject.instance()
    
    if layer_id in project.mapLayers():
        layer = project.mapLayer(layer_id)
        
        if layer.type() != QgsMapLayer.VectorLayer:
            raise Exception(f"Layer is not a vector layer: {layer_id}")
        
        # Build request
        request = QgsFeatureRequest()
        
        # Apply field restriction if specified
        if fields:
            field_indices = []
            for field_name in fields:
                idx = layer.fields().indexFromName(field_name)
                if idx >= 0:
                    field_indices.append(idx)
            
            if field_indices:
                request.setSubsetOfAttributes(field_indices)
        
        # Apply filter expression
        if filter_expression:
            expr = QgsExpression(filter_expression)
            if expr.hasParserError():
                raise Exception(f"Invalid filter expression: {expr.parserErrorString()}")
            request.setFilterExpression(filter_expression)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Export based on format
        format = format.lower()
        if format == 'csv':
            # Export to CSV
            with open(output_path, 'w', newline='') as csvfile:
                # Get field names
                if fields:
                    fieldnames = fields
                else:
                    fieldnames = [field.name() for field in layer.fields()]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write features
                for feature in layer.getFeatures(request):
                    row = {}
                    for field_name in fieldnames:
                        row[field_name] = feature.attribute(field_name)
                    writer.writerow(row)
                
            return {
                "success": True,
                "layer_id": layer_id,
                "format": "csv",
                "output_path": output_path,
                "fields": fieldnames
            }
            
        elif format == 'json':
            # Export to JSON
            features_data = []
            
            # Get field names
            if fields:
                fieldnames = fields
            else:
                fieldnames = [field.name() for field in layer.fields()]
            
            # Write features
            for feature in layer.getFeatures(request):
                feature_data = {}
                for field_name in fieldnames:
                    value = feature.attribute(field_name)
                    feature_data[field_name] = value
                features_data.append(feature_data)
            
            # Write to file
            with open(output_path, 'w') as jsonfile:
                json.dump(features_data, jsonfile, indent=2)
            
            return {
                "success": True,
                "layer_id": layer_id,
                "format": "json",
                "output_path": output_path,
                "fields": fieldnames
            }
            
        else:
            raise Exception(f"Unsupported export format: {format}. Supported formats: csv, json")
    else:
        raise Exception(f"Layer not found: {layer_id}")

class QgisMCPDockWidget(QDockWidget):
    """Dock widget for the QGIS MCP plugin"""
    closed = pyqtSignal()
    
    def __init__(self, iface):
        super().__init__("QGIS MCP")
        self.iface = iface
        self.server = None
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the dock widget UI"""
        # Create widget and layout
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Add port selection
        layout.addWidget(QLabel("Server Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setMinimum(1024)
        self.port_spin.setMaximum(65535)
        self.port_spin.setValue(9876)
        layout.addWidget(self.port_spin)
        
        # Add server control buttons
        self.start_button = QPushButton("Start Server")
        self.start_button.clicked.connect(self.start_server)
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Server")
        self.stop_button.clicked.connect(self.stop_server)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        # Add status label
        self.status_label = QLabel("Server: Stopped")
        layout.addWidget(self.status_label)
        
        # Add to dock widget
        self.setWidget(widget)
    
    def start_server(self):
        """Start the server"""
        if not self.server:
            port = self.port_spin.value()
            self.server = QgisMCPServer(port=port, iface=self.iface)
            
        if self.server.start():
            self.status_label.setText(f"Server: Running on port {self.server.port}")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.port_spin.setEnabled(False)
    
    def stop_server(self):
        """Stop the server"""
        if self.server:
            self.server.stop()
            self.server = None
            
        self.status_label.setText("Server: Stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.port_spin.setEnabled(True)
        
    def closeEvent(self, event):
        """Stop server on dock close"""
        self.stop_server()
        self.closed.emit()
        super().closeEvent(event)


class QgisMCPPlugin:
    """Main plugin class for QGIS MCP"""
    
    def __init__(self, iface):
        self.iface = iface
        self.dock_widget = None
        self.action = None
    
    def initGui(self):
        """Initialize GUI"""
        # Create action
        self.action = QAction(
            "QGIS MCP",
            self.iface.mainWindow()
        )
        self.action.setCheckable(True)
        self.action.triggered.connect(self.toggle_dock)
        
        # Add to plugins menu and toolbar
        self.iface.addPluginToMenu("QGIS MCP", self.action)
        self.iface.addToolBarIcon(self.action)
    
    def toggle_dock(self, checked):
        """Toggle the dock widget"""
        if checked:
            # Create dock widget if it doesn't exist
            if not self.dock_widget:
                self.dock_widget = QgisMCPDockWidget(self.iface)
                self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
                # Connect close event
                self.dock_widget.closed.connect(self.dock_closed)
            else:
                # Show existing dock widget
                self.dock_widget.show()
        else:
            # Hide dock widget
            if self.dock_widget:
                self.dock_widget.hide()
    
    def dock_closed(self):
        """Handle dock widget closed"""
        self.action.setChecked(False)
    
    def unload(self):
        """Unload plugin"""
        # Stop server if running
        if self.dock_widget:
            self.dock_widget.stop_server()
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget = None
            
        # Remove plugin menu item and toolbar icon
        self.iface.removePluginMenu("QGIS MCP", self.action)
        self.iface.removeToolBarIcon(self.action)


# Plugin entry point
def classFactory(iface):
    return QgisMCPPlugin(iface)
